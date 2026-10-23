// Size-independent fused convolutions (CUTLASS 2.x implicit GEMM, sm_80-style mma.sync, fp16 accumulation, EpilogueWithVisitor),
// compiled ONCE into a torch extension -- the problem size is a runtime argument, nothing is compiled or tuned per input size.
// Fast-FS kernels (geo lookup, GRU gates, ...) at the end.
//
//   y = act(conv(x, w) + bias[k] (+ res[n, p, q, k]))        x, res, y: NHWC (= channels-last NCHW) fp16, w: KRSC fp16
//   act: 0 none, 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu(erf), 6 gelu(tanh)
// Requires C % 8 == 0 and K % 8 == 0 (16-byte vectors); dilation 1, groups 1. The caller falls back to torch otherwise.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

struct EpiArgs {
    const __half* bias;   // [K] or null
    const __half* res;    // [M, K] or null
    __half* y;            // [M, ldy] (ldy >= K: output into a channel slice of a wider buffer)
    int act;
    int ldy;
    // optional output index map (sub-pixel transposed convs): row (n, z', p', q') of the sub-conv -> output voxel
    // (n, 2 z' + bz, 2 p' + bp, 2 q' + bq) of an [N, Do, Ho, Wo, ldy] buffer; out-of-range voxels are skipped
    int map = 0, Zs = 1, Ps = 1, Qs = 1, bz = 0, bp = 0, bq = 0, Do = 1, Ho = 1, Wo = 1;
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return v / (1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        case 7: return 1.f / (1.f + __expf(-v));
        case 8: return tanhf(v);
        case 9: return v > 0.f ? v : 0.01f * v;
    }
    return v;
}

template <class OutIt>
struct Visitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};
    static int const kFrags = ThreadMap::Iterations::kCount;

    EpiArgs const& a;
    OutIt it;
    int M, N;
    uint4 pr[kFrags];

    CUTLASS_DEVICE Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                           int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (!a.res) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) pr[i] = __ldg(reinterpret_cast<const uint4*>(a.res + p * N + n));
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        long long dst = p * a.ldy;
        if (a.map) {
            long long t = p;
            int q = t % a.Qs; t /= a.Qs; int pp = t % a.Ps; t /= a.Ps; int z = t % a.Zs; long long nb = t / a.Zs;
            int zo = 2 * z + a.bz, yo = 2 * pp + a.bp, xo = 2 * q + a.bq;
            if (zo < 0 || zo >= a.Do || yo < 0 || yo >= a.Ho || xo < 0 || xo >= a.Wo) return;
            dst = (((nb * a.Do + zo) * a.Ho + yo) * a.Wo + xo) * a.ldy;
        }
        const __half* av = reinterpret_cast<const __half*>(&acc);
        __align__(16) __half bv[8];
        if (a.bias) *reinterpret_cast<uint4*>(bv) = __ldg(reinterpret_cast<const uint4*>(a.bias + n));
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.bias) v += __half2float(bv[k]);
            if (a.res) v += __half2float(rv[k]);
            o[k] = __float2half(act_f(v, a.act));
        }
        *reinterpret_cast<uint4*>(a.y + dst + n) = *reinterpret_cast<const uint4*>(o);
    }
};

template <class TB, class WS, int Stages, int Swz,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, Alg>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                  {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

static int dispatch(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                    cudaStream_t s) {
    switch (cfg) {
        case 0: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 1: return run<Fused<GS<128, 64, 32>, GS<64, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 2: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 3: return run<Fused<GS<64, 128, 32>, GS<32, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 4: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 5: return run<Fused<GS<128, 256, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 6: return run<Fused<GS<256, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 7: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 8: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 9: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        // stem (C == 8, zero-padded RGB): fixed-channels iterator, no wasted K
        case 10: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 11: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 12: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 4, 1, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        // narrow-K (output channel) tiles for mobile nets (K = 16 .. 160)
        case 13: return run<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 14: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 3, 4>>(p, x, w, e, s);
        case 15: return run<Fused<GS<256, 32, 32>, GS<64, 32, 32>, 3, 4>>(p, x, w, e, s);
        case 16: return run<Fused<GS<128, 32, 64>, GS<32, 32, 64>, 3, 4>>(p, x, w, e, s);
        case 17: return run<Fused<GS<128, 96, 32>, GS<32, 32, 32>, 3, 4>>(p, x, w, e, s);
        // large-GEMM tiles with K-step 64 (ViT MLP / qkv as 1x1 convs; sm_120 smem <= 99 KB)
        case 18: return run<Fused<GS<128, 128, 64>, GS<64, 64, 64>, 3, 4>>(p, x, w, e, s);
        case 19: return run<Fused<GS<128, 256, 64>, GS<64, 64, 64>, 2, 4>>(p, x, w, e, s);
        case 20: return run<Fused<GS<256, 128, 64>, GS<64, 64, 64>, 2, 4>>(p, x, w, e, s);
        case 21: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 22: return run<Fused<GS<128, 256, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 23: return run<Fused<GS<256, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 24: return run<Fused<GS<128, 128, 64>, GS<64, 32, 64>, 3, 4>>(p, x, w, e, s);
        case 25: return run<Fused<GS<64, 128, 64>, GS<32, 64, 64>, 3, 4>>(p, x, w, e, s);
    }
    return -1;
}

static const int SMS = [] { int d = 0, n = 36; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }();

// fixed rule (no timing; from a ResNet-50 sweep): 128x128 tiles with 8 warps when they fill the GPU,
// 128x64 / 8 warps for K = 64 (mod 128), smaller tiles when there are too few CTAs (batch 1)
// fixed rule (no timing; from sweeps of ResNet-50 / MobileNet V2 / EfficientNet-B0 convs at batch 1 / 8 / 64):
// 128x128 tiles (8 warps) when K fits 128-wide tiles and they fill the GPU, else 128x64 (8 warps),
// 64x64 when there are too few CTAs (batch 1). cfg 17 (128x96, 12 warps) computes garbage (NaN) -> never picked.
// cuBLAS fallback for tiny-M 1x1 GEMMs: disabled by default (measured slower than the 64x64 CUTLASS tile at batch 1:
// the extra epilogue launches cost more than split-K saves); CLF_FALLBACK_MACS=<n> enables it above n MACs (A/B only)
static const long long FALLBACK_MACS = [] { const char* v = getenv("CLF_FALLBACK_MACS"); return v ? atoll(v) : (1ll << 62); }();

static int pick(long long M, int K, int C, int R, int S) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    auto fits = [&](int tn) { return ((K + tn - 1) / tn) * tn <= 1.1 * K; };   // <= 10 % padded output channels
    if (C == 8) return tiles(128, 64) >= SMS ? 10 : 12;   // 128x128 fixed-channel tiles (cfg 11) measured slower everywhere
    if (fits(128)) {
        if (tiles(128, 128) >= SMS) return 8;
        if (tiles(64, 128) >= SMS) return 7;
        return 2;
    }
    if (tiles(128, 64) >= SMS) return 9;     // also for K not a multiple of 64 (mobile nets): ~= 64x64 3-stage, >> 5-stage
    return 2;
}

static at::Tensor stem_c8(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int ph, int pw, int sh, int sw, int act);

static at::Tensor conv_impl(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                            std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, int64_t cfg,
                            c10::optional<at::Tensor> out) {
    // x [N, C, H, W] channels-last fp16, w [K, C, R, S] channels-last fp16 (= KRSC), res [N, K, P, Q] channels-last;
    // out (optional): [N, K, P, Q] channel slice of a channels-last buffer (pixel stride ldy, 16-byte aligned)
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf, "fp16 CUDA tensors");
    x = x.contiguous(at::MemoryFormat::ChannelsLast);   // no-op when already channels-last (shuffles / splits may not be)
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast), "weight channels-last");
    if (res) res = res->contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int K = w.size(0), R = w.size(2), S = w.size(3);
    int ph = padding[0], pw = padding[1], sh = stride[0], sw = stride[1];
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    at::Tensor y;
    int ldy = K;
    if (out) {
        y = *out;
        ldy = y.stride(3);
        TORCH_CHECK(y.size(0) == N && y.size(1) == K && y.size(2) == P && y.size(3) == Q && y.stride(1) == 1 &&
                    y.stride(2) == (long long)Q * ldy && y.stride(0) == (long long)P * Q * ldy && ldy % 8 == 0 &&
                    ((uintptr_t)y.data_ptr()) % 16 == 0 && y.scalar_type() == at::kHalf, "conv out slice layout");
    } else {
        y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    }
    if (C == 8 && K <= 16 && !res && !out && cfg < 0)   // direct kernel wins only for K <= 16 (smem bank conflicts above)
        return stem_c8(x, w, bias, ph, pw, sh, sw, (int)act);
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{bias ? (const __half*)bias->data_ptr() : nullptr,
              res ? (const __half*)res->data_ptr() : nullptr, (__half*)y.data_ptr(), (int)act, ldy};
    if (res) TORCH_CHECK(res->sizes() == y.sizes(), "res shape");
    if (cfg < 0) {
        long long M = (long long)N * P * Q;
        // 1x1 / stride 1 GEMMs with too few CTAs even for 64x64 tiles (batch 1, late stages): cuBLAS (split-K) + epilogue ops
        // (only when the GEMM is big enough for split-K to matter: otherwise the extra epilogue launches cost more)
        if (R == 1 && S == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 && ((M + 63) / 64) * ((K + 63) / 64) < SMS &&
            M * (long long)K * C >= FALLBACK_MACS) {
            auto x2 = x.permute({0, 2, 3, 1}).reshape({M, C});
            auto w2 = w.view({K, C});
            auto o = bias ? at::addmm(*bias, x2, w2.t()) : at::mm(x2, w2.t());
            if (res) o.add_(res->permute({0, 2, 3, 1}).reshape({M, K}));
            switch (act) {
                case 1: o.relu_(); break;
                case 2: o.clamp_(0, 6); break;
                case 3: o = at::silu(o); break;
                case 4: o = at::hardswish(o); break;
                case 5: o = at::gelu(o); break;
                case 6: o = at::gelu(o, "tanh"); break;
            }
            auto o4 = o.view({N, P, Q, K}).permute({0, 3, 1, 2});
            if (out) { y.copy_(o4); return y; }
            return o4;
        }
        cfg = pick(M, K, C, R, S);
    }
    int rc = dispatch(cfg, p, x.data_ptr(), w.data_ptr(), e, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(rc == 0, "conv_fused failed rc=", rc, " cfg=", cfg, " x=", x.sizes(), " w=", w.sizes());
    return y;
}

at::Tensor conv_fused(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                      std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, int64_t cfg) {
    return conv_impl(x, w, bias, res, padding, stride, act, cfg, c10::nullopt);
}

void conv_fused_into(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                     std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, at::Tensor out, int64_t cfg) {
    conv_impl(x, w, bias, res, padding, stride, act, cfg, out);
}

// y = act(x * scale[c] + shift[c]) for an NHWC fp16 tensor that may be a channel slice (pixel stride ldx); packed output.
// (pre-activation BatchNorm + ReLU of DenseNet / unfoldable BNs)
__global__ void affine_act_k(const __half* __restrict__ x, const float* __restrict__ sc, const float* __restrict__ sh,
                             __half* __restrict__ y, long long npix, int C8, long long ldx, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= npix * C8) return;
    int c = i % C8; long long p = i / C8;
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x + p * ldx) + c);
    const __half* h = reinterpret_cast<const __half*>(&v);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(__half2float(h[k]) * sc[c * 8 + k] + sh[c * 8 + k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor affine_act(at::Tensor x, at::Tensor scale, at::Tensor shift, int64_t act) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    long long ldx = x.stride(3);
    bool slice_ok = x.stride(1) == 1 && x.stride(2) == W * ldx && x.stride(0) == (long long)H * W * ldx && ldx % 8 == 0 &&
                    ((uintptr_t)x.data_ptr()) % 16 == 0;
    if (!slice_ok) { x = x.contiguous(at::MemoryFormat::ChannelsLast); ldx = C; }
    TORCH_CHECK(x.scalar_type() == at::kHalf && C % 8 == 0 && scale.scalar_type() == at::kFloat, "affine_act input");
    auto y = at::empty({N, C, H, W}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long npix = (long long)N * H * W, total = npix * (C / 8);
    affine_act_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), scale.data_ptr<float>(), shift.data_ptr<float>(), (__half*)y.data_ptr(), npix, C / 8, ldx, (int)act);
    return y;
}

// max pool, NHWC fp16, 8 channels per thread (C % 8 == 0), -inf padding (PyTorch semantics, no ceil_mode / dilation)
__global__ void maxpool_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q,
                              int R, int S, int sh, int sw, int ph, int pw) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    __half2 m[4];
    for (int k = 0; k < 4; ++k) m[k] = __half2(__ushort_as_half(0xfc00), __ushort_as_half(0xfc00));
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int w = q * sw - pw + s2; if (w < 0 || w >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + w) * C8 * 8) + c);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
            for (int k = 0; k < 4; ++k) m[k] = __hmax2(m[k], hv[k]);
        }
    }
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(m);
}

at::Tensor maxpool(at::Tensor x, std::vector<int64_t> k, std::vector<int64_t> stride, std::vector<int64_t> pad, bool ceil) {
    if (x.scalar_type() != at::kHalf || x.size(1) % 8) return at::max_pool2d(x, k, stride, pad, {1, 1}, ceil);
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    // PyTorch output size (ceil_mode: last window must start inside the input or left padding)
    auto osz = [&](int L, int kk, int s, int p) {
        int o = ceil ? (L + 2 * p - kk + s - 1) / s + 1 : (L + 2 * p - kk) / s + 1;
        if (ceil && (o - 1) * s >= L + p) --o;
        return o;
    };
    int P = osz(H, k[0], stride[0], pad[0]), Q = osz(W, k[1], stride[1], pad[1]);
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (C / 8);
    maxpool_nhwc8<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (__half*)y.data_ptr(), N, H, W, C / 8, P, Q, k[0], k[1], stride[0], stride[1], pad[0], pad[1]);
    return y;
}

// depthwise conv (groups == C == K), NHWC fp16, weight [R, S, C] fp16, + bias + act; 8 channels of one output pixel per
// thread, fp32 accumulation (memory-bound: accumulation precision is free here)
__global__ void dwconv_nhwc8(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                             __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q, int R, int S, int sh, int sw,
                             int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
    if (b) {
        uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c);
        const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = __half2float(hb[k]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = 0.f;
    }
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * C8 * 8) + c);
            uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
            const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 a = __half22float2(hv[k]), bb = __half22float2(hg[k]);
                acc[2 * k] += a.x * bb.x; acc[2 * k + 1] += a.y * bb.y;
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// register-tiled variant: one thread = 8 channels x TW consecutive output columns; weights and the input row segment are
// loaded once per thread and reused across the TW outputs (fp32 accumulation). R, S <= 7.
template <int TW>
__global__ void dwconv_tw(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q, int R, int S, int sh, int sw,
                          int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int QT = (Q + TW - 1) / TW;
    long long total = (long long)N * P * QT * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int qt = t % QT; t /= QT; int p = t % P; int n = t / P;
    int q0 = qt * TW;
    float acc[TW][8];
    float bb[8];
    if (b) {
        uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c);
        const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
        for (int k = 0; k < 8; ++k) bb[k] = __half2float(hb[k]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) bb[k] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[j][k] = bb[k];
    const int L = (TW - 1) * sw + S;          // input columns touched by the TW outputs
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        const __half* xrow = x + ((long long)n * H + h) * W * C8 * 8;
        for (int l = 0; l < L; ++l) {
            int ww = q0 * sw - pw + l; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(xrow + (long long)ww * C8 * 8) + c);
            float xv[8];
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) { float2 f = __half22float2(hv[k]); xv[2 * k] = f.x; xv[2 * k + 1] = f.y; }
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s2 = l - j * sw;                       // tap index of input column l for output j
                if (s2 < 0 || s2 >= S) continue;
                uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
                const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 f = __half22float2(hg[k]);
                    acc[j][2 * k] += xv[2 * k] * f.x; acc[j][2 * k + 1] += xv[2 * k + 1] * f.y;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        int q = q0 + j; if (q >= Q) break;
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[j][k], act));
        reinterpret_cast<uint4*>(y + (((long long)n * P + p) * Q + q) * C8 * 8)[c] = *reinterpret_cast<uint4*>(o);
    }
}

static const int DW_TW = [] { const char* v = getenv("CLF_DW_TW"); return v ? atoi(v) : 2; }();   // TW 2: 7x7 -17 %, 3x3 ~0 (memory bound)

at::Tensor dwconv(at::Tensor x, at::Tensor w_rsc, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                  std::vector<int64_t> padding, int64_t act) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.size(1) % 8 == 0 && w_rsc.is_contiguous(), "dwconv input");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), R = w_rsc.size(0), S = w_rsc.size(1);
    int P = (H + 2 * padding[0] - R) / stride[0] + 1, Q = (W + 2 * padding[1] - S) / stride[1] + 1;
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* xp = (const __half*)x.data_ptr(); const __half* wp = (const __half*)w_rsc.data_ptr();
    const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr; __half* yp = (__half*)y.data_ptr();
    if (DW_TW == 2 || DW_TW == 4) {
        long long total = (long long)N * P * ((Q + DW_TW - 1) / DW_TW) * (C / 8);
        if (DW_TW == 4)
            dwconv_tw<4><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, stride[0], stride[1], padding[0], padding[1], (int)act);
        else
            dwconv_tw<2><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, stride[0], stride[1], padding[0], padding[1], (int)act);
        return y;
    }
    long long total = (long long)N * P * Q * (C / 8);
    dwconv_nhwc8<<<(total + 255) / 256, 256, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, stride[0], stride[1],
                                                       padding[0], padding[1], (int)act);
    return y;
}

// direct conv for the zero-padded RGB stem (C == 8) with few output channels (K <= 64): one thread = 8 output channels of
// one pixel; weights [K/8][R][S][8 in][8 out] in shared memory (implicit GEMM wastes most of a 64-wide tile at K = 16 / 32)
__global__ void stem_c8_k(const __half* __restrict__ x, const __half* __restrict__ wt, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int K8, int P, int Q, int R, int S, int sh, int sw,
                          int ph, int pw, int act) {
    extern __shared__ __half ws[];
    int nw = K8 * R * S * 64;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) ws[i] = wt[i];
    __syncthreads();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * K8;
    if (i >= total) return;
    int kg = i % K8; long long t = i / K8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = b ? __half2float(b[kg * 8 + k]) : 0.f;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * 8));
            const __half* hv = reinterpret_cast<const __half*>(&v);
            const __half* wp = ws + ((kg * R + r) * S + s2) * 64;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float xv = __half2float(hv[c]);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += xv * __half2float(wp[c * 8 + k]);
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

static at::Tensor stem_c8(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int ph, int pw, int sh, int sw, int act) {
    int N = x.size(0), H = x.size(2), W = x.size(3), K = w.size(0), R = w.size(2), S = w.size(3);
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    // [K, 8, R, S] -> [K/8][R][S][8 in][8 out]
    auto wt = w.contiguous().view({K / 8, 8, 8, R, S}).permute({0, 3, 4, 2, 1}).contiguous();
    auto y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (K / 8);
    int smem = K * R * S * 8 * 2;
    stem_c8_k<<<(total + 127) / 128, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (const __half*)wt.data_ptr(), bias ? (const __half*)bias->data_ptr() : nullptr,
        (__half*)y.data_ptr(), N, H, W, K / 8, P, Q, R, S, sh, sw, ph, pw, act);
    return y;
}

// avg pool NHWC fp16 (count_include_pad as PyTorch), 8 channels per thread
__global__ void avgpool_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q,
                              int R, int S, int sh, int sw, int ph, int pw, bool cip) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int cnt = 0;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * C8 * 8) + c);
            const __half* hv = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] += __half2float(hv[k]);
            ++cnt;
        }
    }
    int h0 = p * sh - ph, w0 = q * sw - pw;
    int hend = min(h0 + R, H + ph), wend = min(w0 + S, W + pw);
    float div = cip ? (float)((hend - h0) * (wend - w0)) : (float)max(cnt, 1);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(acc[k] / div);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor avgpool(at::Tensor x, std::vector<int64_t> k, std::vector<int64_t> stride, std::vector<int64_t> pad, bool cip) {
    if (x.scalar_type() != at::kHalf || x.size(1) % 8) return at::avg_pool2d(x, k, stride, pad, false, cip);
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int P = (H + 2 * pad[0] - k[0]) / stride[0] + 1, Q = (W + 2 * pad[1] - k[1]) / stride[1] + 1;
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (C / 8);
    avgpool_nhwc8<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (__half*)y.data_ptr(), N, H, W, C / 8, P, Q, k[0], k[1], stride[0], stride[1], pad[0], pad[1], cip);
    return y;
}

// network input: fp32 / fp16 NCHW (any strides via contiguous) -> fp16 NHWC with channels zero-padded to 8 (stem conv C 3 -> 8)
template <class T>
__global__ void input_nhwc8_k(const T* __restrict__ x, __half* __restrict__ y, int C, long long HW, long long total) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;   // one output pixel (8 channels) per thread
    if (i >= total) return;
    long long n = i / HW, hw = i % HW;
    __align__(16) __half o[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) o[c] = c < C ? __float2half((float)x[(n * C + c) * HW + hw]) : __float2half(0.f);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor input_nhwc8(at::Tensor x) {
    x = x.contiguous();
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C <= 8, "input_nhwc8: C <= 8");
    auto y = at::empty({N, 8, H, W}, x.options().dtype(at::kHalf).memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * H * W;
    auto st = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kFloat)
        input_nhwc8_k<float><<<(total + 255) / 256, 256, 0, st>>>(x.data_ptr<float>(), (__half*)y.data_ptr(), C, (long long)H * W, total);
    else
        input_nhwc8_k<__half><<<(total + 255) / 256, 256, 0, st>>>((const __half*)x.data_ptr(), (__half*)y.data_ptr(), C, (long long)H * W, total);
    return y;
}


// ---------------------------------------------------------------- fused Conv3d (NDHWC, CUTLASS implicit GEMM, fp16 accumulate)
using NDHWC = cutlass::layout::TensorNDHWC;
template <class TB, class WS, int Stages, int Swz>
struct Fused3 {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;
    using Base = typename cutlass::conv::kernel::DefaultConv3dFprop<
        Hf, NDHWC, Hf, NDHWC, Hf, NDHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<V, typename Base::Epilogue>::Epilogue;
    union SharedStorage { typename Mma::SharedStorage main_loop; typename Epilogue::SharedStorage epilogue; };
    struct Params { typename Base::Params base; EpiArgs e; };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv3_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    auto const& ps = prm.problem_size;
    cutlass::MatrixCoord extent(ps.N * ps.Z * ps.P * ps.Q, ps.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run3(cutlass::conv::Conv3dProblemSize const& p, const void* x, const void* w, EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    typename Base::Arguments args(p, {(Hf*)x, NDHWC::packed({p.N, p.D, p.H, p.W, p.C})}, {(Hf*)w, NDHWC::packed({p.K, p.T, p.R, p.S, p.C})},
                                  {nullptr, NDHWC::packed({p.N, p.Z, p.P, p.Q, p.K})}, {nullptr, NDHWC::packed({p.N, p.Z, p.P, p.Q, p.K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv3_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 2;
        attr_set = true;
    }
    fused_conv3_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

static int dispatch3(int cfg, cutlass::conv::Conv3dProblemSize const& p, const void* x, const void* w, EpiArgs const& e, cudaStream_t s) {
    switch (cfg) {
        case 0: return run3<Fused3<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 1: return run3<Fused3<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 2: return run3<Fused3<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 3: return run3<Fused3<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 4: return run3<Fused3<GS<256, 32, 32>, GS<64, 32, 32>, 3, 4>>(p, x, w, e, s);
    }
    return -1;
}

// fixed rule (no timing): output-channel tile 32 for K <= 32, 64 for K <= 64 (or K % 128 > 64-wide waste), else 128x128;
// 64x64 when there are too few CTAs
static int pick3(long long M, int K) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    if (K <= 32) return tiles(128, 32) >= SMS ? 3 : 2;
    bool fit128 = ((K + 127) / 128) * 128 <= 1.1 * K;
    if (fit128 && tiles(128, 128) >= SMS) return 0;
    if (tiles(128, 64) >= SMS) return 1;
    return 2;
}

at::Tensor conv3_impl(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                      std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, int64_t cfg, c10::optional<at::Tensor> out,
                      std::vector<int64_t> bmap = {}) {
    // x [N, C, D, H, W] channels-last-3d fp16, w [K, C, T, R, S] channels-last-3d, res like y; out: channel slice (ldy) or none
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && x.dim() == 5, "fp16 5-D CUDA tensors");
    x = x.contiguous(at::MemoryFormat::ChannelsLast3d);
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast3d), "weight channels-last-3d");
    if (res) res = res->contiguous(at::MemoryFormat::ChannelsLast3d);
    int N = x.size(0), C = x.size(1), D = x.size(2), H = x.size(3), W = x.size(4);
    int K = w.size(0), T = w.size(2), R = w.size(3), S = w.size(4);
    TORCH_CHECK(C % 8 == 0 && K % 8 == 0, "conv3_fused: C, K % 8");
    int pd = padding[0], ph = padding[1], pw = padding[2], sd = stride[0], sh = stride[1], sw = stride[2];
    int Z = (D + 2 * pd - T) / sd + 1, P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    at::Tensor y; int ldy = K;
    if (!bmap.empty()) {   // sub-pixel map into the full transposed-conv output `out` [N, K, Do, Ho, Wo] channels-last-3d
        TORCH_CHECK(out && !res && out->is_contiguous(at::MemoryFormat::ChannelsLast3d) && out->size(1) == K && out->size(0) == N, "conv3 map out");
        y = *out;
    } else if (out) {
        y = *out; ldy = y.stride(4);
        TORCH_CHECK(y.size(0) == N && y.size(1) == K && y.size(2) == Z && y.size(3) == P && y.size(4) == Q && y.stride(1) == 1 &&
                    y.stride(3) == (long long)Q * ldy && y.stride(2) == (long long)P * Q * ldy && y.stride(0) == (long long)Z * P * Q * ldy &&
                    ldy % 8 == 0 && ((uintptr_t)y.data_ptr()) % 16 == 0, "conv3 out slice layout");
    } else {
        y = at::empty({N, K, Z, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast3d));
    }
    if (res) TORCH_CHECK(res->sizes() == y.sizes(), "res shape");
    if (!bmap.empty()) TORCH_CHECK(bmap.size() == 3, "bmap: 3 offsets");
    cutlass::conv::Conv3dProblemSize p(N, D, H, W, C, K, T, R, S, Z, P, Q, pd, ph, pw, sd, sh, sw, 1, 1, 1,
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{bias ? (const __half*)bias->data_ptr() : nullptr, res ? (const __half*)res->data_ptr() : nullptr,
              (__half*)y.data_ptr(), (int)act, ldy};
    if (!bmap.empty()) {
        e.map = 1; e.Zs = Z; e.Ps = P; e.Qs = Q; e.bz = bmap[0]; e.bp = bmap[1]; e.bq = bmap[2];
        e.Do = y.size(2); e.Ho = y.size(3); e.Wo = y.size(4);
    }
    if (cfg < 0) cfg = pick3((long long)N * Z * P * Q, K);
    int rc = dispatch3(cfg, p, x.data_ptr(), w.data_ptr(), e, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(rc == 0, "conv3_fused failed rc=", rc, " cfg=", cfg, " x=", x.sizes(), " w=", w.sizes());
    return y;
}

at::Tensor conv3_fused(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                       std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, int64_t cfg) {
    return conv3_impl(x, w, bias, res, padding, stride, act, cfg, c10::nullopt);
}
void conv3_fused_into(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                      std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, at::Tensor out) {
    conv3_impl(x, w, bias, res, padding, stride, act, -1, out);
}
// one parity class of a stride-2 transposed Conv3d (sub-pixel decomposition): 2x2x2 correlation with padding 1 written to
// output voxels (2 z' + b) (b in {0, -1} per dim) of the full output tensor
void conv3_subpixel_into(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int64_t act, at::Tensor out, std::vector<int64_t> b) {
    conv3_impl(x, w, bias, c10::nullopt, {1, 1, 1}, {1, 1, 1}, act, -1, out, b);
}

// ------------------------------------------------------------------------------------------------ Fast-FS kernels
// Combined geometry lookup (== GeoFast: 1-D linear interpolation with zero padding along D of the geometry volume and along
// W of the all-pairs correlation, 2 levels) written directly in the NHWC layout of the padded convc1 input:
//   [geo l0 (k, c) | geo l1 (k, c) | corr l0 (k) | corr l1 (k) | zero pad]   (k = tap 0..2r, c = 0..Cg-1)
// One thread per (pixel, level, tap). Also writes disp (fp16) into an 8-channel buffer (convd1 input) and a channel of mf.
__global__ void geo_lookup_k(const __half* __restrict__ g0, const __half* __restrict__ g1, const __half* __restrict__ c0,
                             const __half* __restrict__ c1, const float* __restrict__ disp, __half* __restrict__ out,
                             __half* __restrict__ d8, __half* __restrict__ mf, int B, int H, int W, int D0, int D1, int W0,
                             int W1, int Cg, int Cw, int K, int ldo, int ldmf, int mfoff, int fly) {
    // fly: level 1 computed from level 0 on the fly (pairwise means along D / W, rounded to fp16 like a stored pooled level)
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long P = (long long)B * H * W;
    if (i >= P * 2 * K) return;
    int k = i % K; int l = (i / K) % 2; long long p = i / (2 * K);
    int r = K / 2;
    long long hw = p % ((long long)H * W); int b = p / ((long long)H * W); int w = hw % W;
    float d = disp[p];
    float s = l ? 0.5f : 1.f;
    // geometry volume
    int Dn = l ? D1 : D0;
    const __half* g = l ? g1 : g0;
    float pos = d * s + (float)(k - r);
    float x0 = floorf(pos); float f = pos - x0; int i0 = (int)x0, i1 = i0 + 1;
    float w0 = (i0 >= 0 && i0 <= Dn - 1) ? 1.f - f : 0.f;
    float w1 = (i1 >= 0 && i1 <= Dn - 1) ? f : 0.f;
    i0 = min(max(i0, 0), Dn - 1); i1 = min(max(i1, 0), Dn - 1);
    __half* o = out + p * ldo + (long long)(l * K + k) * Cw;
    if (l == 1 && fly) {
        const __half* r0 = g0 + (((long long)b * D0 + 2 * i0) * H * W + hw) * Cg;
        const __half* r1 = g0 + (((long long)b * D0 + 2 * i1) * H * W + hw) * Cg;
        long long st = (long long)H * W * Cg;
        for (int c = 0; c < Cw; c += 4) {
            uint2 a = __ldg(reinterpret_cast<const uint2*>(r0 + c)), a2 = __ldg(reinterpret_cast<const uint2*>(r0 + st + c));
            uint2 bb = __ldg(reinterpret_cast<const uint2*>(r1 + c)), b2 = __ldg(reinterpret_cast<const uint2*>(r1 + st + c));
            const __half* ah = reinterpret_cast<const __half*>(&a); const __half* ah2 = reinterpret_cast<const __half*>(&a2);
            const __half* bh = reinterpret_cast<const __half*>(&bb); const __half* bh2 = reinterpret_cast<const __half*>(&b2);
            __align__(8) __half v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float pa = __half2float(__float2half(0.5f * (__half2float(ah[j]) + __half2float(ah2[j]))));
                float pb = __half2float(__float2half(0.5f * (__half2float(bh[j]) + __half2float(bh2[j]))));
                v[j] = __float2half(pa * w0 + pb * w1);
            }
            *reinterpret_cast<uint2*>(o + c) = *reinterpret_cast<uint2*>(v);
        }
    } else {
        const __half* r0 = g + (((long long)b * Dn + i0) * H * W + hw) * Cg;
        const __half* r1 = g + (((long long)b * Dn + i1) * H * W + hw) * Cg;
        for (int c = 0; c < Cw; c += 4) {
            uint2 a = __ldg(reinterpret_cast<const uint2*>(r0 + c)), bb = __ldg(reinterpret_cast<const uint2*>(r1 + c));
            const __half* ah = reinterpret_cast<const __half*>(&a); const __half* bh = reinterpret_cast<const __half*>(&bb);
            __align__(8) __half v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = __float2half(__half2float(ah[j]) * w0 + __half2float(bh[j]) * w1);
            *reinterpret_cast<uint2*>(o + c) = *reinterpret_cast<uint2*>(v);
        }
    }
    // correlation
    int Wn = l ? W1 : W0;
    pos = (float)w * s - d * s + (float)(k - r);
    x0 = floorf(pos); f = pos - x0; i0 = (int)x0; i1 = i0 + 1;
    w0 = (i0 >= 0 && i0 <= Wn - 1) ? 1.f - f : 0.f;
    w1 = (i1 >= 0 && i1 <= Wn - 1) ? f : 0.f;
    i0 = min(max(i0, 0), Wn - 1); i1 = min(max(i1, 0), Wn - 1);
    float cv0, cv1;
    if (l == 1 && fly) {
        const __half* cc = c0 + p * W0;
        cv0 = __half2float(__float2half(0.5f * (__half2float(cc[2 * i0]) + __half2float(cc[2 * i0 + 1]))));
        cv1 = __half2float(__float2half(0.5f * (__half2float(cc[2 * i1]) + __half2float(cc[2 * i1 + 1]))));
    } else {
        const __half* cc = (l ? c1 : c0) + p * Wn;
        cv0 = __half2float(cc[i0]); cv1 = __half2float(cc[i1]);
    }
    out[p * ldo + 2 * K * Cw + l * K + k] = __float2half(cv0 * w0 + cv1 * w1);
    if (l == 0 && k == 0) {
        __half dh = __float2half(d);
        if (d8) d8[p * 8] = dh;
        if (mf) mf[p * ldmf + mfoff] = dh;
    }
}

void geo_lookup(at::Tensor g0, at::Tensor g1, at::Tensor c0, at::Tensor c1, at::Tensor disp, at::Tensor out,
                c10::optional<at::Tensor> d8, c10::optional<at::Tensor> mf, int64_t K, int64_t mfoff, int64_t Cw) {
    // Cw: channels used per geometry entry (<= row stride Cg; padded volumes)
    // g*: [B, D, H, W, Cg] fp16 contiguous; c*: [B, H, W, W*] fp16 contiguous; disp [B, H, W] fp32; out [B, H, W, ldo]
    TORCH_CHECK(g0.is_contiguous() && g1.is_contiguous() && c0.is_contiguous() && c1.is_contiguous() && disp.is_contiguous()
                && out.is_contiguous() && disp.scalar_type() == at::kFloat, "geo_lookup layout");
    int B = g0.size(0), D0 = g0.size(1), H = g0.size(2), W = g0.size(3), Cg = g0.size(4);
    int fly = g1.numel() == 0 ? 1 : 0;   // empty level-1 tensors: pooled on the fly from level 0
    int D1 = fly ? D0 / 2 : g1.size(1);
    if (Cw <= 0) Cw = Cg;
    TORCH_CHECK(Cg % 4 == 0 && Cw % 4 == 0 && Cw <= Cg, "Cg, Cw % 4");
    int W0 = c0.size(3), W1 = fly ? W0 / 2 : c1.size(3), ldo = out.size(3);
    TORCH_CHECK(ldo >= 2 * K * (Cw + 1), "out channels");
    int ldmf = mf ? mf->size(3) : 0;
    long long total = (long long)B * H * W * 2 * K;
    geo_lookup_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)g0.data_ptr(), (const __half*)g1.data_ptr(), (const __half*)c0.data_ptr(), (const __half*)c1.data_ptr(),
        disp.data_ptr<float>(), (__half*)out.data_ptr(), d8 ? (__half*)d8->data_ptr() : nullptr,
        mf ? (__half*)mf->data_ptr() : nullptr, B, H, W, D0, D1, W0, W1, Cg, (int)Cw, (int)K, ldo, ldmf, (int)mfoff, fly);
}

// GRU step 1: q inputs [r*h | x] for the small (1x1) and large (3x3) GRU. zr*: [P, 2*Hd] (z | r, sigmoid applied), h [P, Hd],
// hx [P, ldhx] (x in channels 0..Cx-1), qs / ql [P, Hd + Cx]. 8 channels per thread.
__global__ void gru_rh_k(const __half* __restrict__ zrs, const __half* __restrict__ zrl, const __half* __restrict__ h,
                         const __half* __restrict__ hx, __half* __restrict__ qs, __half* __restrict__ ql, long long P, int Hd,
                         int Cx, int ldhx) {
    int n8 = (Hd + Cx) / 8;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= P * n8) return;
    int c = (i % n8) * 8; long long p = i / n8;
    int ldq = Hd + Cx;
    if (c < Hd) {
        uint4 hv = __ldg(reinterpret_cast<const uint4*>(h + p * Hd + c));
        uint4 rs = __ldg(reinterpret_cast<const uint4*>(zrs + p * 2 * Hd + Hd + c));
        uint4 rl = __ldg(reinterpret_cast<const uint4*>(zrl + p * 2 * Hd + Hd + c));
        __half2* a = reinterpret_cast<__half2*>(&hv); __half2* bs = reinterpret_cast<__half2*>(&rs); __half2* bl = reinterpret_cast<__half2*>(&rl);
        __align__(16) __half2 os[4], ol[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { os[j] = __hmul2(a[j], bs[j]); ol[j] = __hmul2(a[j], bl[j]); }
        *reinterpret_cast<uint4*>(qs + p * ldq + c) = *reinterpret_cast<uint4*>(os);
        *reinterpret_cast<uint4*>(ql + p * ldq + c) = *reinterpret_cast<uint4*>(ol);
    } else {
        uint4 xv = __ldg(reinterpret_cast<const uint4*>(hx + p * ldhx + c - Hd));
        *reinterpret_cast<uint4*>(qs + p * ldq + c) = xv;
        *reinterpret_cast<uint4*>(ql + p * ldq + c) = xv;
    }
}

void gru_rh(at::Tensor zrs, at::Tensor zrl, at::Tensor h, at::Tensor hx, at::Tensor qs, at::Tensor ql, int64_t Cx) {
    long long P = h.numel() / h.size(1);
    int Hd = h.size(1), ldhx = hx.size(1);
    TORCH_CHECK(Hd % 8 == 0 && Cx % 8 == 0 && qs.size(1) == Hd + Cx, "gru_rh dims");
    long long total = P * ((Hd + Cx) / 8);
    gru_rh_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)zrs.data_ptr(), (const __half*)zrl.data_ptr(), (const __half*)h.data_ptr(), (const __half*)hx.data_ptr(),
        (__half*)qs.data_ptr(), (__half*)ql.data_ptr(), P, Hd, (int)Cx, ldhx);
}

// GRU step 2: h' = att * ((1-zs) h + zs qs) + (1-att) * ((1-zl) h + zl ql)  (qs / ql: tanh applied); writes h (in place) and
// the h channels of hx (offset hxoff).
__global__ void gru_out_k(const __half* __restrict__ zrs, const __half* __restrict__ zrl, const __half* __restrict__ qs,
                          const __half* __restrict__ ql, const __half* __restrict__ att, __half* __restrict__ h,
                          __half* __restrict__ hx, long long P, int Hd, int ldhx, int hxoff) {
    int n8 = Hd / 8;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= P * n8) return;
    int c = (i % n8) * 8; long long p = i / n8;
    float a = __half2float(att[p]);
    uint4 hv = *reinterpret_cast<const uint4*>(h + p * Hd + c);
    uint4 zs = __ldg(reinterpret_cast<const uint4*>(zrs + p * 2 * Hd + c));
    uint4 zl = __ldg(reinterpret_cast<const uint4*>(zrl + p * 2 * Hd + c));
    uint4 vs = __ldg(reinterpret_cast<const uint4*>(qs + p * Hd + c));
    uint4 vl = __ldg(reinterpret_cast<const uint4*>(ql + p * Hd + c));
    const __half* H_ = reinterpret_cast<const __half*>(&hv); const __half* ZS = reinterpret_cast<const __half*>(&zs);
    const __half* ZL = reinterpret_cast<const __half*>(&zl); const __half* QS = reinterpret_cast<const __half*>(&vs);
    const __half* QL = reinterpret_cast<const __half*>(&vl);
    __align__(16) __half o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float hh = __half2float(H_[j]);
        float s_ = __half2float(__float2half((1.f - __half2float(ZS[j])) * hh + __half2float(ZS[j]) * __half2float(QS[j])));
        float l_ = __half2float(__float2half((1.f - __half2float(ZL[j])) * hh + __half2float(ZL[j]) * __half2float(QL[j])));
        o[j] = __float2half(s_ * a + l_ * (1.f - a));
    }
    *reinterpret_cast<uint4*>(h + p * Hd + c) = *reinterpret_cast<uint4*>(o);
    *reinterpret_cast<uint4*>(hx + p * ldhx + hxoff + c) = *reinterpret_cast<uint4*>(o);
}

void gru_out(at::Tensor zrs, at::Tensor zrl, at::Tensor qs, at::Tensor ql, at::Tensor att, at::Tensor h, at::Tensor hx,
             int64_t hxoff) {
    long long P = h.numel() / h.size(1);
    int Hd = h.size(1), ldhx = hx.size(1);
    long long total = P * (Hd / 8);
    gru_out_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)zrs.data_ptr(), (const __half*)zrl.data_ptr(), (const __half*)qs.data_ptr(), (const __half*)ql.data_ptr(),
        (const __half*)att.data_ptr(), (__half*)h.data_ptr(), (__half*)hx.data_ptr(), P, Hd, ldhx, (int)hxoff);
}

// Combined cost volume (= cat([gwc volume, concat volume], 1) of Fast-FS, NDHWC): out[b, d, h, w, :] =
//   [ sum_c nl[b,h,w,g*Cg+c] * nr[b,h,w-d,g*Cg+c] (g < G) | L[b,h,w,:] | R[b,h,w-d,:] ]   (w - d < 0: gwc and R parts 0)
// nl / nr: features (norm = 1: group-normalized in the kernel, F.normalize semantics) [B, H, W, G*Cg] fp16; L / R: [B, H, W, Cc] fp16. fp32 accumulation.
// Block = (b, h, tile of TW columns); left / right feature rows staged in shared memory.
template <int TW>
__global__ void build_volume_k(const __half* __restrict__ nl, const __half* __restrict__ nr, const __half* __restrict__ L,
                               const __half* __restrict__ R, __half* __restrict__ out, int B, int H, int W, int D, int G, int Cg,
                               int Cc, int norm) {
    extern __shared__ __half sm[];
    int CF = G * Cg;
    int w0 = blockIdx.x * TW; int h = blockIdx.y; int b = blockIdx.z;
    int nr_cols = TW + D - 1;              // right columns w0-D+1 .. w0+TW-1
    __half* sl = sm;                       // [TW][CF]
    __half* sr = sm + TW * CF;             // [nr_cols][CF]
    long long rowbase = ((long long)b * H + h) * W;
    for (int i = threadIdx.x; i < TW * CF / 2; i += blockDim.x) {
        int wl = (i * 2) / CF, c = (i * 2) % CF; int w = w0 + wl;
        __half2 v = w < W ? *reinterpret_cast<const __half2*>(nl + (rowbase + w) * CF + c) : __float2half2_rn(0.f);
        *reinterpret_cast<__half2*>(sl + wl * CF + c) = v;
    }
    for (int i = threadIdx.x; i < nr_cols * CF / 2; i += blockDim.x) {
        int wr = (i * 2) / CF, c = (i * 2) % CF; int w = w0 - D + 1 + wr;
        __half2 v = (w >= 0 && w < W) ? *reinterpret_cast<const __half2*>(nr + (rowbase + w) * CF + c) : __float2half2_rn(0.f);
        *reinterpret_cast<__half2*>(sr + wr * CF + c) = v;
    }
    __syncthreads();
    if (norm) {   // F.normalize over each group's channels (fp32), rounded to fp16 (= upstream before the product)
        for (int j = threadIdx.x; j < (TW + nr_cols) * G; j += blockDim.x) {
            int row = j / G, g = j % G;
            __half* v = (row < TW ? sl + row * CF : sr + (row - TW) * CF) + g * Cg;
            float ss = 0.f;
            for (int c = 0; c < Cg; ++c) { float f = __half2float(v[c]); ss += f * f; }
            float inv = 1.f / fmaxf(sqrtf(ss), 1e-12f);
            for (int c = 0; c < Cg; ++c) v[c] = __float2half(__half2float(v[c]) * inv);
        }
        __syncthreads();
    }
    int Ctot = G + 2 * Cc;
    for (int j = threadIdx.x; j < TW * D; j += blockDim.x) {
        int wl = j % TW, d = j / TW; int w = w0 + wl;
        if (w >= W) continue;
        __half* o = out + ((((long long)b * D + d) * H + h) * W + w) * Ctot;
        bool valid = w - d >= 0;
        const __half* a = sl + wl * CF;
        const __half* r = sr + (wl - d + D - 1) * CF;
        __align__(16) __half ob[64];
        for (int g = 0; g < G; ++g) {
            float acc = 0.f;
            if (valid) {
                for (int c = 0; c < Cg; c += 2) {
                    float2 x = __half22float2(*reinterpret_cast<const __half2*>(a + g * Cg + c));
                    float2 y = __half22float2(*reinterpret_cast<const __half2*>(r + g * Cg + c));
                    acc += x.x * y.x + x.y * y.y;
                }
            }
            ob[g] = __float2half(acc);
        }
        const __half* lp = L + (rowbase + w) * Cc;
        const __half* rp = R + (rowbase + w - d) * Cc;
        for (int c = 0; c < Cc; ++c) {
            ob[G + c] = lp[c];
            ob[G + Cc + c] = valid ? rp[c] : __float2half(0.f);
        }
        for (int c = 0; c < Ctot; c += 8) *reinterpret_cast<uint4*>(o + c) = *reinterpret_cast<const uint4*>(ob + c);
    }
}

at::Tensor build_volume(at::Tensor nl, at::Tensor nr, at::Tensor L, at::Tensor R, int64_t D, int64_t G, int64_t norm) {
    TORCH_CHECK(nl.is_contiguous() && nr.is_contiguous() && L.is_contiguous() && R.is_contiguous(), "build_volume: contiguous [B,H,W,C]");
    int B = nl.size(0), H = nl.size(1), W = nl.size(2), CF = nl.size(3), Cc = L.size(3);
    int Cg = CF / G;
    TORCH_CHECK(Cg % 2 == 0 && CF == G * Cg, "build_volume: group channels even");
    TORCH_CHECK((G + 2 * Cc) % 8 == 0 && G + 2 * Cc <= 64, "build_volume: output channels % 8, <= 64");
    int Ctot = G + 2 * Cc;
    auto out = at::empty({B, Ctot, D, H, W}, nl.options().memory_format(at::MemoryFormat::ChannelsLast3d));
    constexpr int TW = 32;
    size_t smem = (size_t)(TW + TW + D - 1) * CF * 2;
    static size_t attr = 0;
    if (smem > attr) {
        TORCH_CHECK(cudaFuncSetAttribute(build_volume_k<TW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                    "build_volume: shared memory ", smem);
        attr = smem;
    }
    dim3 grid((W + TW - 1) / TW, H, B);
    build_volume_k<TW><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)nl.data_ptr(), (const __half*)nr.data_ptr(), (const __half*)L.data_ptr(), (const __half*)R.data_ptr(),
        (__half*)out.data_ptr(), B, H, W, (int)D, (int)G, Cg, Cc, (int)norm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// context_upsample(disp_low * scale, softmax(logits, 1)) of Fast-FS (= unfold 3x3 zero-pad, nearest x4 upsample, weighted sum):
// logits [B, 9, H4, W4] any strides (fp16), disp [B, h, w] fp32 -> out [B, H4, W4] fp32; low pixel = (y * h / H4, x * w / W4).
__global__ void ctx_up_k(const __half* __restrict__ lg, long long sb, long long sc, long long sy, long long sx,
                         const float* __restrict__ disp, float* __restrict__ out, int B, int H4, int W4, int h, int w, float scale) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)B * H4 * W4) return;
    int x = i % W4; int y = (i / W4) % H4; int b = i / ((long long)W4 * H4);
    int yl = (int)(((long long)y * h) / H4), xl = (int)(((long long)x * w) / W4);
    const __half* l = lg + b * sb + y * sy + x * sx;
    float v[9], mx = -1e30f;
#pragma unroll
    for (int k = 0; k < 9; ++k) { v[k] = __half2float(l[k * sc]); mx = fmaxf(mx, v[k]); }
    float se = 0.f, acc = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        float e = __expf(v[k] - mx); se += e;
        int yy = yl + k / 3 - 1, xx = xl + k % 3 - 1;
        float dv = (yy >= 0 && yy < h && xx >= 0 && xx < w) ? disp[((long long)b * h + yy) * w + xx] : 0.f;
        acc += e * dv;
    }
    out[i] = acc / se * scale;
}

at::Tensor ctx_up(at::Tensor logits, at::Tensor disp, double scale) {
    TORCH_CHECK(logits.scalar_type() == at::kHalf && logits.size(1) == 9 && disp.is_contiguous() && disp.scalar_type() == at::kFloat, "ctx_up");
    int B = logits.size(0), H4 = logits.size(2), W4 = logits.size(3), h = disp.size(1), w = disp.size(2);
    auto out = at::empty({B, H4, W4}, disp.options());
    long long n = (long long)B * H4 * W4;
    ctx_up_k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)logits.data_ptr(), logits.stride(0),
        logits.stride(1), logits.stride(2), logits.stride(3), disp.data_ptr<float>(), out.data_ptr<float>(), B, H4, W4, h, w, (float)scale);
    return out;
}

// y = trilinear_upsample(x, size of r, align_corners=False) + r   (NDHWC fp16, C % 8 == 0; PyTorch source-index rule)
__device__ __forceinline__ void src_idx(int o, int in, int out, int& i0, int& i1, float& l) {
    float sc = (float)in / (float)out;
    float s = fmaxf(sc * (o + 0.5f) - 0.5f, 0.f);
    i0 = min((int)s, in - 1); i1 = min(i0 + 1, in - 1); l = s - (float)i0;
}
__global__ void up3_add_k(const __half* __restrict__ x, const __half* __restrict__ r, __half* __restrict__ y, int B, int C8,
                          int Di, int Hi, int Wi, int Do, int Ho, int Wo) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long tot = (long long)B * Do * Ho * Wo * C8;
    if (i >= tot) return;
    int c = i % C8; long long p = i / C8;
    int w = p % Wo; int h = (p / Wo) % Ho; int d = (p / ((long long)Wo * Ho)) % Do; int b = p / ((long long)Wo * Ho * Do);
    int d0, d1, h0, h1, w0, w1; float ld, lh, lw;
    src_idx(d, Di, Do, d0, d1, ld); src_idx(h, Hi, Ho, h0, h1, lh); src_idx(w, Wi, Wo, w0, w1, lw);
    float acc[8];
    uint4 rv = __ldg(reinterpret_cast<const uint4*>(r) + i);
    const __half* rh = reinterpret_cast<const __half*>(&rv);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        int dd = (t & 4) ? d1 : d0, hh = (t & 2) ? h1 : h0, ww = (t & 1) ? w1 : w0;
        float wt = ((t & 4) ? ld : 1.f - ld) * ((t & 2) ? lh : 1.f - lh) * ((t & 1) ? lw : 1.f - lw);
        uint4 v = __ldg(reinterpret_cast<const uint4*>(x + ((((long long)b * Di + dd) * Hi + hh) * Wi + ww) * C8 * 8) + c);
        const __half* vh = reinterpret_cast<const __half*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += wt * __half2float(vh[k]);
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(__half2float(__float2half(acc[k])) + __half2float(rh[k]));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor up3_add(at::Tensor x, at::Tensor r) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast3d); r = r.contiguous(at::MemoryFormat::ChannelsLast3d);
    int B = x.size(0), C = x.size(1);
    TORCH_CHECK(C % 8 == 0 && r.size(1) == C && x.scalar_type() == at::kHalf && r.scalar_type() == at::kHalf, "up3_add");
    auto y = at::empty_like(r, r.options().memory_format(at::MemoryFormat::ChannelsLast3d));
    long long tot = r.numel() / 8;
    up3_add_k<<<(tot + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)x.data_ptr(), (const __half*)r.data_ptr(),
        (__half*)y.data_ptr(), B, C / 8, x.size(2), x.size(3), x.size(4), r.size(2), r.size(3), r.size(4));
    return y;
}

// classifier tail: cost[d] = bias + o[d][1] + o[d-1][0] + o[d+1][2] (the 3 disparity taps of a 3x3x3 conv computed as output channels of
// one 2-D conv, o: [B, D, H, W, 8] fp16), then softmax over d and soft-argmax: disp[b, h, w] = sum_d softmax(cost)_d * d (fp32)
__global__ void cls_regress_k(const __half* __restrict__ o, float* __restrict__ disp, int B, int D, long long HW, float bias) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= B * HW) return;
    long long b = i / HW, p = i % HW;
    const uint4* base = reinterpret_cast<const uint4*>(o) + b * D * HW + p;
    float mx = -1e30f, se = 0.f, acc = 0.f;
    uint4 prev, cur = __ldg(base), nxt;
    for (int d = 0; d < D; ++d) {
        if (d + 1 < D) nxt = __ldg(base + (long long)(d + 1) * HW);
        const __half* c = reinterpret_cast<const __half*>(&cur);
        float v = bias + __half2float(c[1]);
        if (d > 0) v += __half2float(reinterpret_cast<const __half*>(&prev)[0]);
        if (d + 1 < D) v += __half2float(reinterpret_cast<const __half*>(&nxt)[2]);
        if (v > mx) { float sc = __expf(mx - v); se *= sc; acc *= sc; mx = v; }
        float e = __expf(v - mx); se += e; acc += e * d;
        prev = cur; cur = nxt;
    }
    disp[i] = acc / se;
}

at::Tensor cls_regress(at::Tensor o, int64_t B, int64_t D, double bias) {
    // o: [B*D, 8, H, W] channels-last fp16 (= [B, D, H, W, 8])
    TORCH_CHECK(o.is_contiguous(at::MemoryFormat::ChannelsLast) && o.size(1) == 8 && o.scalar_type() == at::kHalf, "cls_regress input");
    int H = o.size(2), W = o.size(3);
    auto disp = at::empty({B, H, W}, o.options().dtype(at::kFloat));
    long long n = B * (long long)H * W;
    cls_regress_k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)o.data_ptr(), disp.data_ptr<float>(),
        (int)B, (int)D, (long long)H * W, (float)bias);
    return disp;
}

// LayerNorm over the last dim of a contiguous [R, C] fp16 tensor (C % 8 == 0, C <= 512): 8 lanes per row, fp32 statistics
// (two-pass in registers), affine; = F.layer_norm. PyTorch's kernel uses one block per row (slow for C = 48 .. 304).
__global__ void ln_rows_k(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                          __half* __restrict__ y, long long R, int C, float eps, int l2) {
    // l2: y = x / max(||x||_2, eps) (= F.normalize), no affine
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 8;
    int lane = threadIdx.x % 8;
    bool ok = row < R;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (ok ? row : 0) * C);
    int n8 = C / 8;
    float v[64];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c8 = lane + j * 8;
        if (c8 < n8 && ok) {
            uint4 q = __ldg(xr + c8);
            const __half* h = reinterpret_cast<const __half*>(&q);
#pragma unroll
            for (int t = 0; t < 8; ++t) { v[j * 8 + t] = __half2float(h[t]); s += v[j * 8 + t]; }
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) v[j * 8 + t] = 0.f;
        }
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o, 8);
    float mean = l2 ? 0.f : s / C, q2 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c8 = lane + j * 8;
        if (c8 < n8) {
#pragma unroll
            for (int t = 0; t < 8; ++t) { float d = v[j * 8 + t] - mean; q2 += d * d; }
        }
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) q2 += __shfl_xor_sync(0xffffffff, q2, o, 8);
    float rs = l2 ? 1.f / fmaxf(sqrtf(q2), eps) : rsqrtf(q2 / C + eps);
    if (!ok) return;
    if (l2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c8 = lane + j * 8;
            if (c8 < n8) {
                __align__(16) __half o[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) o[t] = __float2half(v[j * 8 + t] * rs);
                reinterpret_cast<uint4*>(y + row * C)[c8] = *reinterpret_cast<uint4*>(o);
            }
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c8 = lane + j * 8;
        if (c8 < n8) {
            uint4 wq = __ldg(reinterpret_cast<const uint4*>(w) + c8), bq = __ldg(reinterpret_cast<const uint4*>(b) + c8);
            const __half* wh = reinterpret_cast<const __half*>(&wq); const __half* bh = reinterpret_cast<const __half*>(&bq);
            __align__(16) __half o[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) o[t] = __float2half((v[j * 8 + t] - mean) * rs * __half2float(wh[t]) + __half2float(bh[t]));
            reinterpret_cast<uint4*>(y + row * C)[c8] = *reinterpret_cast<uint4*>(o);
        }
    }
}

at::Tensor ln_rows(at::Tensor x, c10::optional<at::Tensor> w_, c10::optional<at::Tensor> b_, double eps) {
    // w / b None: L2 normalization (F.normalize) instead of LayerNorm
    int l2 = !w_.has_value();
    TORCH_CHECK(x.is_contiguous() && x.dim() == 2 && x.scalar_type() == at::kHalf, "ln_rows input");
    at::Tensor w = l2 ? x : *w_, b = l2 ? x : *b_;
    long long R = x.size(0); int C = x.size(1);
    TORCH_CHECK(C % 8 == 0 && C <= 512, "ln_rows: C % 8, <= 512");
    auto y = at::empty_like(x);
    long long th = R * 8;
    ln_rows_k<<<(th + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)x.data_ptr(), (const __half*)w.data_ptr(),
        (const __half*)b.data_ptr(), (__half*)y.data_ptr(), R, C, (float)eps, l2);
    return y;
}

TORCH_LIBRARY(ffs, m) {
    m.def("ln_rows(Tensor x, Tensor? w, Tensor? b, float eps) -> Tensor");
    m.def("cls_regress(Tensor o, int B, int D, float bias) -> Tensor");
    m.def("up3_add(Tensor x, Tensor r) -> Tensor");
    m.def("ctx_up(Tensor logits, Tensor disp, float scale) -> Tensor");
    m.def("build_volume(Tensor nl, Tensor nr, Tensor L, Tensor R, int D, int G, int norm=0) -> Tensor");
    m.def("conv3_fused(Tensor x, Tensor w, Tensor? bias, Tensor? res, int[] padding, int[] stride, int act, int cfg=-1) -> Tensor");
    m.def("conv3_subpixel_into(Tensor x, Tensor w, Tensor? bias, int act, Tensor(a!) out, int[] b) -> ()");
    m.def("conv3_fused_into(Tensor x, Tensor w, Tensor? bias, Tensor? res, int[] padding, int[] stride, int act, Tensor(a!) out) -> ()");
    m.def("geo_lookup(Tensor g0, Tensor g1, Tensor c0, Tensor c1, Tensor disp, Tensor(a!) out, Tensor(b!)? d8, Tensor(c!)? mf, int K, int mfoff, int Cw=0) -> ()");
    m.def("gru_rh(Tensor zrs, Tensor zrl, Tensor h, Tensor hx, Tensor(a!) qs, Tensor(b!) ql, int Cx) -> ()");
    m.def("gru_out(Tensor zrs, Tensor zrl, Tensor qs, Tensor ql, Tensor att, Tensor(a!) h, Tensor(b!) hx, int hxoff) -> ()");
    m.def("input_nhwc8(Tensor x) -> Tensor");
    m.def("conv_fused_into(Tensor x, Tensor w, Tensor? bias, Tensor? res, int[] padding, int[] stride, int act, Tensor(a!) out, int cfg=-1) -> ()");
    m.def("affine_act(Tensor x, Tensor scale, Tensor shift, int act) -> Tensor");
    m.def("avgpool(Tensor x, int[] k, int[] stride, int[] pad, bool count_include_pad=True) -> Tensor");
    m.def("dwconv(Tensor x, Tensor w_rsc, Tensor? bias, int[] stride, int[] padding, int act) -> Tensor");
    m.def("maxpool(Tensor x, int[] k, int[] stride, int[] pad, bool ceil=False) -> Tensor");
    m.def("conv_fused(Tensor x, Tensor w, Tensor? bias, Tensor? res, int[] padding, int[] stride, int act, int cfg=-1) -> Tensor");
}
TORCH_LIBRARY_IMPL(ffs, CUDA, m) { m.impl("conv_fused", conv_fused); m.impl("maxpool", maxpool); m.impl("input_nhwc8", input_nhwc8); m.impl("dwconv", dwconv); m.impl("conv_fused_into", conv_fused_into); m.impl("affine_act", affine_act); m.impl("avgpool", avgpool);
    m.impl("geo_lookup", geo_lookup); m.impl("ln_rows", ln_rows); m.impl("cls_regress", cls_regress); m.impl("up3_add", up3_add); m.impl("ctx_up", ctx_up); m.impl("build_volume", build_volume); m.impl("conv3_fused", conv3_fused); m.impl("conv3_fused_into", conv3_fused_into); m.impl("conv3_subpixel_into", conv3_subpixel_into); m.impl("gru_rh", gru_rh); m.impl("gru_out", gru_out); }

