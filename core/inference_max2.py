"""Fast-FS max path v2 (custom CUDA kernels, core/csrc/ffs_fused.cu): what TensorRT did better than the plain fast path
(core/inference_patch.py), as general (any size) prebuilt kernels.

    from core.inference_max2 import prepare_max2, forward_max2
    prepare_max2(model); disp = forward_max2(model, L, R, iters=8)

v1 -> v2:
  * the distilled network has irregular channel counts (28, 60, 97, 100, 522, 36, 212, ...): cuDNN pads every input (nhwcAddPadding)
    and picks slow kernels. v2 keeps every GRU / classifier activation zero-padded to a multiple of 8 channels; weights are padded
    once at prepare time (exact: padded channels carry zeros / are multiplied by zero weights).
  * all GRU convs = CUTLASS implicit GEMM with fp16 accumulation and fused bias + activation (relu / sigmoid / tanh / gelu) and the
    EdgeNeXt residual, writing straight into channel slices of the concat buffers (no torch.cat, no separate bias / act kernels);
    z|r convs merged (one launch per GRU branch).
  * geometry lookup: one kernel (2 levels x 9 taps, geo volume + all-pairs correlation) writing the padded convc1 input.
  * GRU gates: two elementwise kernels (r*h | x copies; selective h update).
  * mask head only after the last iteration (v1 computed it in every iteration; TRT drops the unused ones).
  * classifier Conv3d(28 -> 1, 3x3x3) as one 2-D conv (the 3 disparity taps as output channels) + shifted sum.
  * fixed per-layer CUTLASS tile configs for a few GRU convs (measured best at KITTI and Midd-H alike;
    no runtime tuning, independent of the input size).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import submodule
import core.foundation_stereo as fsmod
from core import inference_patch as ip
from core import ffs_ext

ops = None


def p8(c):
    return (c + 7) // 8 * 8


def _wb(mod):
    """(weight [K, C, R, S] fp32, bias [K] fp32 or None) of a Conv2d or Linear."""
    w = mod.weight.detach().float()
    if w.dim() == 2:
        w = w[:, :, None, None]
    b = mod.bias.detach().float() if mod.bias is not None else None
    return w, b


class PConv:
    """2-D conv with zero-padded channels through ffs.conv_fused. srcs: list of (module or (w, b), out_src0, n_out, out_dst0);
    in_segs: list of (in_src0, n_in, in_dst0); cin / kout: padded channel counts."""

    def __init__(self, srcs, in_segs, cin, kout, padding, act=0, scale=None, dev="cuda", cfg=-1):
        first = srcs[0][0]
        w0, _ = first if isinstance(first, tuple) else _wb(first)
        R, S = w0.shape[2:]
        W = torch.zeros(kout, cin, R, S)
        B = torch.zeros(kout)
        for mod, o0, no, od in srcs:
            w, b = mod if isinstance(mod, tuple) else _wb(mod)
            w = w.cpu(); b = b.cpu() if b is not None else None
            if scale is not None:
                w = w * scale.view(-1, 1, 1, 1).cpu(); b = b * scale.cpu() if b is not None else None
            for i0, ni, idst in in_segs:
                W[od:od + no, idst:idst + ni] = w[o0:o0 + no, i0:i0 + ni]
            if b is not None:
                B[od:od + no] = b[o0:o0 + no]
        self.w = W.half().to(dev).contiguous(memory_format=torch.channels_last)
        self.b = B.half().to(dev)
        self.pad = list(padding); self.act = act; self.cfg = cfg

    def __call__(self, x, res=None, out=None):
        if out is None:
            return ops.conv_fused(x, self.w, self.b, res, self.pad, [1, 1], self.act, self.cfg)
        ops.conv_fused_into(x, self.w, self.b, res, self.pad, [1, 1], self.act, out, self.cfg)
        return out


def _nchw(buf, B, H, W, c0=None, c1=None):
    """[P, C] (or [B, H, W, C]) buffer -> [B, c, H, W] channels-last view of channels c0:c1."""
    v = buf.view(B, H, W, -1)
    if c0 is not None:
        v = v[..., c0:c1]
    return v.permute(0, 3, 1, 2)


def prepare_max2(model):
    global ops
    ffs_ext.load()
    ops = torch.ops.ffs
    ip.prepare_fast(model, hacc=True)   # BN folded, fp16, channels-last, convc1 input permuted to the GeoFast order
    m = model
    ub = m.update_block
    enc, gru, dh = ub.encoder, ub.gru04, ub.disp_head
    C, K, L = m.volume_dim, 2 * m.args.corr_radius + 1, m.args.corr_levels
    assert L == 2, "v2 lookup kernel: 2 levels"
    hd = gru.small_gru.convz.out_channels           # 60
    Hd = p8(hd)
    cx = gru.conv0[0].out_channels                   # 100
    Cx = p8(cx)
    c_inp = m.cam.fc[0].in_channels                   # 48 (cnet inp channels)
    c_mo = enc.conv.out_channels                     # 48
    ncorr = L * K * (C + 1)                          # 522
    s = m._v2 = type("V2", (), {})()
    s.C, s.K, s.Hd, s.hd, s.Cx, s.cx, s.c_inp, s.c_mo = C, K, Hd, hd, Cx, cx, c_inp, c_mo
    s.Ccorr = p8(ncorr)
    g = C * K
    # GeoFast order per level: [geo k*C+c (g) | corr (K)]; kernel order: [geo l0 | geo l1 | corr l0 | corr l1]
    s.convc1 = PConv([(enc.convc1, 0, enc.convc1.out_channels, 0)],
                     [(0, g, 0), (g + K, g, g), (g, K, 2 * g), (2 * g + K, K, 2 * g + K)], s.Ccorr, p8(enc.convc1.out_channels), (0, 0), act=1)
    c2 = enc.convc2.out_channels                     # 96
    d2 = enc.convd2.out_channels                     # 12
    s.c2, s.Cenc = c2, p8(c2) + p8(d2)
    s.convc2 = PConv([(enc.convc2, 0, c2, 0)], [(0, enc.convc2.in_channels, 0)], p8(enc.convc2.in_channels), p8(c2), (1, 1), act=1, cfg=7)
    s.convd1 = PConv([(enc.convd1, 0, enc.convd1.out_channels, 0)], [(0, 1, 0)], 8, p8(enc.convd1.out_channels), (3, 3), act=1)
    s.convd2 = PConv([(enc.convd2, 0, d2, 0)], [(0, enc.convd2.in_channels, 0)], p8(enc.convd2.in_channels), p8(d2), (1, 1), act=1)
    s.conv = PConv([(enc.conv, 0, c_mo, 0)], [(0, c2, 0), (c2, d2, p8(c2))], s.Cenc, p8(c_mo), (1, 1), act=1)
    # motion features [inp | out | disp | pad]
    s.Cmf = p8(c_inp + c_mo + 1)
    s.conv0 = PConv([(gru.conv0[0], 0, cx, 0)], [(0, c_inp + c_mo + 1, 0)], s.Cmf, Cx, (1, 1), act=1, cfg=8)
    # hx = [x | h]
    c1o = gru.conv1[0].out_channels                  # 168
    s.Chx, s.Chx2 = Cx + Hd, p8(c1o)
    s.conv1 = PConv([(gru.conv1[0], 0, c1o, 0)], [(0, cx, 0), (cx, hd, Cx)], s.Chx, s.Chx2, (1, 1), act=1)
    zr_in = [(0, gru.small_gru.convz.in_channels, 0)]
    s.zr_s = PConv([(gru.small_gru.convz, 0, hd, 0), (gru.small_gru.convr, 0, hd, Hd)], zr_in, s.Chx2, 2 * Hd, (0, 0), act=7, cfg=7)
    s.zr_l = PConv([(gru.large_gru.convz, 0, hd, 0), (gru.large_gru.convr, 0, hd, Hd)], zr_in, s.Chx2, 2 * Hd, (1, 1), act=7)
    q_in = [(0, hd, 0), (hd, cx, Hd)]
    s.q_s = PConv([(gru.small_gru.convq, 0, hd, 0)], q_in, Hd + Cx, Hd, (0, 0), act=8)
    s.q_l = PConv([(gru.large_gru.convq, 0, hd, 0)], q_in, Hd + Cx, Hd, (1, 1), act=8)
    # disp head
    seq = dh.conv
    c36 = seq[0].out_channels
    s.Cdh = p8(c36)
    s.dh0 = PConv([(seq[0], 0, c36, 0)], [(0, hd, 0)], Hd, s.Cdh, (1, 1), act=1)
    s.enx = []
    for blk in (seq[2], seq[3]):
        assert isinstance(blk.norm, nn.Identity)
        dw = torch.zeros(blk.dwconv.kernel_size[0], blk.dwconv.kernel_size[1], s.Cdh)
        dw[:, :, :c36] = blk.dwconv.weight.detach().float()[:, 0].permute(1, 2, 0)
        dwb = torch.zeros(s.Cdh); dwb[:c36] = blk.dwconv.bias.detach().float()
        e = blk.pwconv1.out_features
        gam = blk.gamma.detach().float() if blk.gamma is not None else None
        pw1 = PConv([(blk.pwconv1, 0, e, 0)], [(0, c36, 0)], s.Cdh, p8(e), (0, 0), act=5, cfg=7)
        pw2 = PConv([(blk.pwconv2, 0, c36, 0)], [(0, e, 0)], p8(e), s.Cdh, (0, 0), act=0, scale=gam)
        s.enx.append((dw.half().cuda().contiguous(), dwb.half().cuda(), blk.dwconv.padding[0], pw1, pw2))
    s.dh_out = PConv([(seq[4], 0, 1, 0)], [(0, c36, 0)], s.Cdh, 8, (1, 1), act=0, cfg=13)
    # classifier Conv3d(C -> 1, 3x3x3): the 3 disparity taps as output channels of one 2-D conv
    cls = m.classifier.layers[0] if hasattr(m.classifier, "layers") else m.classifier[0]
    wc = cls.weight.detach().float()                  # [1, C, 3, 3, 3]
    assert wc.shape[0] == 1 and tuple(wc.shape[2:]) == (3, 3, 3)
    s.Ccls = p8(C)
    s.nfused3 = fuse_cost_path(m, s.Ccls)
    s.nedgenext = fuse_edgenext(m.feature)
    s.nln = fuse_layernorms(m.feature)
    s.cls = PConv([((wc[:, :, kd], None), 0, 1, kd) for kd in range(3)], [(0, C, 0)], s.Ccls, 8, (1, 1), act=0)
    s.cls_b = float(cls.bias.detach().float()[0]) if cls.bias is not None else 0.0
    return model


# ------------------------------------------------------------------------------------- cost volume path (3-D convs)
def _pad_conv_(mod, cin, cout):
    """zero-pad a (BN-folded, fp16) Conv3d / ConvTranspose3d / Conv2d to cin input / cout output channels in place."""
    w = mod.weight.data.float()
    tr = isinstance(mod, (nn.ConvTranspose2d, nn.ConvTranspose3d))
    if tr:   # [Cin, Cout, ...]
        W = w.new_zeros((cin, cout) + tuple(w.shape[2:])); W[:w.shape[0], :w.shape[1]] = w
    else:
        W = w.new_zeros((cout, cin) + tuple(w.shape[2:])); W[:w.shape[0], :w.shape[1]] = w
    mf = torch.channels_last_3d if W.dim() == 5 else torch.channels_last
    mod.weight.data = W.half().contiguous(memory_format=mf)
    if mod.bias is not None:
        b = mod.bias.data.float(); B = b.new_zeros(cout); B[:b.numel()] = b
        mod.bias = nn.Parameter(B.half())
    if tr:
        mod.in_channels, mod.out_channels = cin, cout
    else:
        mod.in_channels, mod.out_channels = cin, cout


def _c3(conv, act, res=None):
    def f(x):
        return ops.conv3_fused(x, conv.weight, conv.bias, res, list(conv.padding), list(conv.stride), act)
    return f


def _fusable3(c):
    return type(c) is nn.Conv3d and c.groups == 1 and all(d == 1 for d in c.dilation) and c.padding_mode == "zeros" \
        and c.in_channels % 8 == 0 and c.out_channels % 8 == 0


def _subpixel_ok(c):
    return type(c) is nn.ConvTranspose3d and c.groups == 1 and all(v == 2 for v in c.stride) and all(v == 1 for v in c.padding) \
        and all(k in (3, 4) for k in c.kernel_size) and all(d == 1 for d in c.dilation) \
        and all(op == (1 if k == 3 else 0) for op, k in zip(c.output_padding, c.kernel_size)) \
        and c.in_channels % 8 == 0 and c.out_channels % 8 == 0


def _subpixel_fwd(c, act):
    """stride-2 ConvTranspose3d (k 4 / p 1, or k 3 / p 1 / output_padding 1) as 8 parity-class 2x2x2 CUTLASS convs (fp16 acc,
    bias + act fused) writing straight into their output voxels (sub-pixel decomposition, same math)."""
    w = c.weight.detach().float()            # [Cin, Cout, kd, kh, kw]
    Ks = c.kernel_size
    subs = []
    for rd in (0, 1):
        for rh in (0, 1):
            for rw in (0, 1):
                W = w.new_zeros(c.out_channels, c.in_channels, 2, 2, 2)
                for jd in (0, 1):
                    for jh in (0, 1):
                        for jw in (0, 1):
                            kd, kh, kw = 3 - rd - 2 * jd, 3 - rh - 2 * jh, 3 - rw - 2 * jw   # tap = 2 - r - 2j + padding
                            if 0 <= kd < Ks[0] and 0 <= kh < Ks[1] and 0 <= kw < Ks[2]:
                                W[:, :, jd, jh, jw] = w[:, :, kd, kh, kw].t()
                subs.append((W.half().contiguous(memory_format=torch.channels_last_3d), [-rd, -rh, -rw]))
    bias = c.bias.detach().half().contiguous() if c.bias is not None else None

    def f(x):
        N, C, D, H, W_ = x.shape
        Do, Ho, Wo = [(n - 1) * 2 - 2 + k + op for n, k, op in zip((D, H, W_), Ks, c.output_padding)]
        x = x.contiguous(memory_format=torch.channels_last_3d)
        out = torch.empty((N, c.out_channels, Do, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        for Wsub, b in subs:
            ops.conv3_subpixel_into(x, Wsub, bias, act, out, b)
        return out
    return f


def fuse_cost_path(m, Cv):
    """(1) zero-pad every volume_dim (28) channel dimension of the cost-volume path to Cv (32); (2) every Conv3d -> CUTLASS
    fused conv3d (fp16 acc, bias + relu / leaky-relu / residual fused)."""
    C = m.volume_dim
    nfused = 0
    roots = [m.corr_stem, m.corr_feature_att, m.cost_agg]
    for root in roots:
        for mod in root.modules():
            if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)) and mod.groups == 1:
                ci = Cv if mod.in_channels == C else mod.in_channels
                co = Cv if mod.out_channels == C else mod.out_channels
                if (ci, co) != (mod.in_channels, mod.out_channels):
                    _pad_conv_(mod, ci, co)
            if isinstance(mod, submodule.FeatureAtt):
                last = mod.feat_att[1]
                if last.out_channels == C:
                    _pad_conv_(last, last.in_channels, Cv)
            if isinstance(mod, submodule.CostVolumeDisparityAttention) and not hasattr(mod, "_v2"):
                mod._v2 = True
                fw = mod.forward
                def att_fwd(cv, window_size=(-1, -1), fw=fw):
                    y = fw(cv[:, :C].contiguous(memory_format=torch.channels_last_3d), window_size)
                    return F.pad(y.permute(0, 2, 3, 4, 1), (0, Cv - C)).permute(0, 4, 1, 2, 3)
                mod.forward = att_fwd
    for root in roots:
        for mod in root.modules():
            if isinstance(mod, submodule.BasicConv) and _subpixel_ok(mod.conv):
                lk = mod.relu is True or isinstance(mod.relu, nn.LeakyReLU)
                mod.forward = _subpixel_fwd(mod.conv, 9 if lk else 0); nfused += 1
                continue
            if isinstance(mod, submodule.BasicConv) and _fusable3(mod.conv):
                assert isinstance(mod.bn, nn.Identity)
                lk = mod.relu is True or isinstance(mod.relu, nn.LeakyReLU)
                if isinstance(mod.relu, nn.LeakyReLU):
                    assert mod.relu.negative_slope == 0.01
                mod.forward = _c3(mod.conv, 9 if lk else 0); nfused += 1
            elif isinstance(mod, submodule.Conv3dNormActReduced):
                for sq in (mod.conv1, mod.conv2):
                    if _fusable3(sq[0]):
                        assert isinstance(sq[1], nn.Identity) and isinstance(sq[2], nn.ReLU)
                        sq.forward = _c3(sq[0], 1); nfused += 1
            elif isinstance(mod, submodule.ResnetBasicBlock3D) and _fusable3(mod.conv1) and _fusable3(mod.conv2):
                assert mod.downsample is None and mod.stride == 1
                def rb(x, mod=mod):
                    y = ops.conv3_fused(x, mod.conv1.weight, mod.conv1.bias, None, list(mod.conv1.padding), [1, 1, 1], 1)
                    return ops.conv3_fused(y, mod.conv2.weight, mod.conv2.bias, x.contiguous(memory_format=torch.channels_last_3d),
                                           list(mod.conv2.padding), [1, 1, 1], 1)
                mod.forward = rb; nfused += 2
    from core.distill_block import PostForwardHelper
    for mod in m.cost_agg.modules():
        if isinstance(mod, PostForwardHelper) and mod.op == "sum" and len(mod.upsample) and isinstance(mod.upsample[-1], nn.Upsample) \
                and mod.upsample[-1].mode == "trilinear" and not mod.upsample[-1].align_corners and mod.upsample[-1].scale_factor is not None:
            def pf(conv2, conv3, left_feat=None, mod=mod):
                y = conv3
                for l in list(mod.upsample)[:-1]:
                    y = l(y)
                sf = float(mod.upsample[-1].scale_factor)
                if y.shape[1] % 8 == 0 and all(o == i * sf for o, i in zip(conv2.shape[2:], y.shape[2:])):
                    x = ops.up3_add(y, conv2)     # trilinear upsample + sum in one pass
                else:
                    x = mod.upsample[-1](y) + conv2
                for layer in mod.out:
                    x = layer(x, left_feat) if isinstance(layer, submodule.FeatureAtt) else layer(x)
                return x
            mod.forward = pf
    return nfused


def _edgenext_block_fwd(blk, x):
    """timm EdgeNeXt ConvBlock: dwconv -> LayerNorm (NHWC) -> fc1 + GELU -> fc2 * gamma + shortcut, the MLP as CUTLASS fused
    1x1 convs (GELU / layer scale + residual in the epilogue)."""
    x = x.contiguous(memory_format=torch.channels_last)
    if blk._v2_dw is not None:
        w, b = blk._v2_dw
        y = ops.dwconv(x, w, b, [1, 1], [w.shape[0] // 2, w.shape[1] // 2], 0)
    else:
        y = blk.conv_dw(x).contiguous(memory_format=torch.channels_last)
    sc = y if blk.shortcut_after_dw else x
    t = _ln(y.permute(0, 2, 3, 1), blk.norm).permute(0, 3, 1, 2)
    return blk._v2_fc2(blk._v2_fc1(t), res=sc)


def _ln(x, norm):
    """LayerNorm over the last dim through ffs.ln_rows (contiguous rows, C % 8 == 0), else F.layer_norm."""
    C = x.shape[-1]
    if x.is_contiguous() and C % 8 == 0 and C <= 512 and x.dtype == torch.half and norm.weight is not None and norm.bias is not None:
        return ops.ln_rows(x.reshape(-1, C), norm.weight, norm.bias, norm.eps).view(x.shape)
    return F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)


def fuse_layernorms(feature):
    n = 0
    for mod in feature.modules():
        if type(mod).__name__ == "LayerNorm2d":
            mod.forward = (lambda x, mod=mod: _ln(x.permute(0, 2, 3, 1), mod).permute(0, 3, 1, 2)); n += 1
        elif type(mod) is nn.LayerNorm:
            mod.forward = (lambda x, mod=mod: _ln(x, mod)); n += 1
    return n


def fuse_edgenext(feature):
    n = 0
    st = getattr(feature, "stem", None)
    if st is not None and type(st[0]) is nn.Conv2d and st[0].in_channels <= 8 and st[0].groups == 1 and st[0].out_channels % 8 == 0:
        c0 = st[0]   # patchify stem (3 -> 48, k4 s4): input zero-padded to 8 channels, CUTLASS fixed-channel conv
        pc = PConv([(c0, 0, c0.out_channels, 0)], [(0, c0.in_channels, 0)], 8, c0.out_channels, c0.padding)
        def stem_fwd(x, pc=pc, stride=list(c0.stride)):
            return ops.conv_fused(ops.input_nhwc8(x), pc.w, pc.b, None, pc.pad, stride, 0, -1)
        c0.forward = stem_fwd
    for blk in feature.modules():
        if type(blk).__name__ != "ConvBlock" or not isinstance(blk.drop_path, nn.Identity):
            continue
        c = blk.norm.normalized_shape[0]
        e = blk.mlp.fc1.out_features
        if c % 8 or e % 8 or not isinstance(blk.mlp.norm, nn.Identity) or not isinstance(blk.mlp.act, nn.GELU) \
                or blk.mlp.act.approximate != "none":
            continue
        dw = blk.conv_dw
        blk._v2_dw = None
        if dw.groups == dw.in_channels == dw.out_channels and dw.stride == (1, 1) and dw.padding_mode == "zeros" \
                and tuple(dw.padding) == (dw.kernel_size[0] // 2, dw.kernel_size[1] // 2):
            blk._v2_dw = (dw.weight.detach()[:, 0].permute(1, 2, 0).contiguous().half(),
                          (dw.bias.detach() if dw.bias is not None else torch.zeros(c, device=dw.weight.device)).half())
        blk._v2_fc1 = PConv([(blk.mlp.fc1, 0, e, 0)], [(0, c, 0)], c, e, (0, 0), act=5, cfg=9)   # 128x64 tiles measured best
        g = blk.gamma.detach().float() if blk.gamma is not None else None
        blk._v2_fc2 = PConv([(blk.mlp.fc2, 0, c, 0)], [(0, e, 0)], e, c, (0, 0), act=0, scale=g)
        blk.forward = (lambda x, blk=blk: _edgenext_block_fwd(blk, x))
        n += 1
    return n


class GeoV2:
    """level-0 geometry volume (the cost-aggregated volume itself, NDHWC view, no copy) + level-0 all-pairs correlation; level 1
    of both is pooled on the fly inside the lookup kernel (no pooled copies)."""

    def __init__(self, fl, fr, comb):
        B, C, H, W = fl.shape
        self.geo = [comb.permute(0, 2, 3, 4, 1), comb.new_empty(0)]
        assert self.geo[0].is_contiguous()
        nrm = lambda f: (ops.ln_rows(f.permute(0, 2, 3, 1).reshape(-1, C), None, None, 1e-12) if f.is_contiguous(memory_format=torch.channels_last)
                         and C % 8 == 0 else F.normalize(f.float(), dim=1).to(comb.dtype).permute(0, 2, 3, 1)).reshape(B * H, W, C)
        f1, f2 = nrm(fl), nrm(fr)   # F.normalize(f.float(), dim=1).half() in one kernel
        self.corr = [torch.bmm(f1, f2.transpose(1, 2)).view(B, H, W, W), comb.new_empty(0)]


def volume_v2(m, f1, f2, D):
    """cat([gwc volume, concat volume]) in one kernel (NDHWC, channels-last-3d [B, G + 2 Cc, D, H, W])."""
    B, Cf, H, W = f1.shape
    G = m.cv_group

    L = m.proj_cmb(f1).permute(0, 2, 3, 1).contiguous()
    R = m.proj_cmb(f2).permute(0, 2, 3, 1).contiguous()
    # group normalization (upstream: F.normalize over the group channels in fp32, rounded to fp16) inside the kernel
    return ops.build_volume(f1.permute(0, 2, 3, 1).contiguous(), f2.permute(0, 2, 3, 1).contiguous(), L, R, D, G,
                            1 if m.args.normalize else 0)


def cam_v2(cam, x):
    """ChannelAttentionEnhancement with plain reductions (adaptive max pool's kernel is slow at 1/4 res)."""
    return torch.sigmoid(cam.fc(x.mean((2, 3), keepdim=True)) + cam.fc(x.amax((2, 3), keepdim=True)))


def classifier_v2(s, comb):
    """comb [B, C, D, H, W] channels-last-3d fp16 -> soft-argmax disparity [B, H, W] fp32
    (= disparity_regression(softmax(Conv3d(C, 1, 3, pad 1)(comb) + bias)))."""
    B, C, D, H, W = comb.shape
    x = comb.permute(0, 2, 3, 4, 1)                  # [B, D, H, W, C] (contiguous for channels-last-3d)
    if C != s.Ccls:
        x = F.pad(x, (0, s.Ccls - C))
    x = x.reshape(B * D, H, W, s.Ccls).permute(0, 3, 1, 2)
    return ops.cls_regress(s.cls(x), B, D, s.cls_b)

def update_v2(s, st, geo, disp):
    """one GRU iteration on the persistent buffers st; returns delta [P] fp32."""
    B, H, W = st["shape"]
    nc = lambda buf, c0=None, c1=None: _nchw(buf, B, H, W, c0, c1)
    ops.geo_lookup(geo.geo[0], geo.geo[1], geo.corr[0], geo.corr[1], disp, st["corr"].view(B, H, W, -1), st["d8"].view(B, H, W, 8),
                   st["mf"].view(B, H, W, -1), s.K, s.c_inp + s.c_mo, s.C)
    c1 = s.convc1(nc(st["corr"]))
    s.convc2(c1, out=nc(st["enc"], 0, p8(s.c2)))
    d1 = s.convd1(nc(st["d8"]))
    s.convd2(d1, out=nc(st["enc"], p8(s.c2), s.Cenc))
    _conv_into_mf(s, st, nc)
    s.conv0(nc(st["mf"]), out=nc(st["hx"], 0, s.Cx))
    hx2 = s.conv1(nc(st["hx"]))
    zrs = s.zr_s(hx2)
    zrl = s.zr_l(hx2)
    zrs_f, zrl_f = zrs.permute(0, 2, 3, 1).reshape(-1, 2 * s.Hd), zrl.permute(0, 2, 3, 1).reshape(-1, 2 * s.Hd)
    ops.gru_rh(zrs_f, zrl_f, st["h"], st["hx"], st["qs"], st["ql"], s.Cx)
    qs = s.q_s(nc(st["qs"])).permute(0, 2, 3, 1).reshape(-1, s.Hd)
    ql = s.q_l(nc(st["ql"])).permute(0, 2, 3, 1).reshape(-1, s.Hd)
    ops.gru_out(zrs_f, zrl_f, qs, ql, st["att"], st["h"], st["hx"], s.Cx)
    t = s.dh0(nc(st["h"]))
    for dw, dwb, pd, pw1, pw2 in s.enx:
        u = ops.dwconv(t, dw, dwb, [1, 1], [pd, pd], 0)
        t = pw2(pw1(u), res=t)
    return s.dh_out(t).permute(0, 2, 3, 1)[..., 0].reshape(B, H, W).float()


def _conv_into_mf(s, st, nc):
    # conv output (c_mo real channels, p8 padded) goes to mf[:, c_inp : c_inp + c_mo]; the padded outputs would overwrite the
    # disp channel, so write into a scratch buffer when c_mo is not a multiple of 8, else straight into the slice
    if s.c_mo % 8 == 0 and s.c_inp % 8 == 0:
        s.conv(nc(st["enc"]), out=nc(st["mf"], s.c_inp, s.c_inp + s.c_mo))
    else:
        o = s.conv(nc(st["enc"]))
        st["mf"].view(*st["shape"], -1)[..., s.c_inp:s.c_inp + s.c_mo] = o.permute(0, 2, 3, 1)[..., :s.c_mo]


@torch.no_grad()
def forward_max2(m, image1, image2, iters=8):
    s = m._v2
    i1, i2 = fsmod.normalize_image(image1), fsmod.normalize_image(image2)
    cl = lambda x: x.half().contiguous(memory_format=ip._mf(x))
    fl = m.feature(cl(i1))     # (one pass with both views as batch 2 measured slower)
    fr = m.feature(cl(i2))
    stem_2x = m.stem_2(cl(i1))
    D = m.args.max_disp // 4
    comb = volume_v2(m, fl[0], fr[0], D)
    comb = m.corr_stem(comb)
    comb = m.corr_feature_att(comb, fl[0])
    comb = m.cost_agg(comb, fl)
    comb = comb.contiguous(memory_format=torch.channels_last_3d)
    disp = classifier_v2(s, comb)                       # [B, H, W] fp32 soft-argmax
    cnet = list(m.cnet(fl[0], fl[1], fl[2]))
    net = torch.tanh(cnet[0][0])
    inp = torch.relu(cnet[0][1])
    inp = cam_v2(m.cam, inp) * inp
    att = m.sam(inp)
    geo = GeoV2(fl[0], fr[0], comb)
    del comb, fl, fr, cnet
    B, _, H, W = net.shape
    P = B * H * W
    dev = net.device
    z = lambda c: torch.zeros(P, c, device=dev, dtype=torch.half)
    st = dict(shape=(B, H, W), corr=z(s.Ccorr), d8=z(8), enc=z(s.Cenc), mf=z(s.Cmf), hx=z(s.Chx), h=z(s.Hd),
              qs=z(s.Hd + s.Cx), ql=z(s.Hd + s.Cx), att=att.reshape(P).contiguous())
    st["h"][:, :s.hd] = net.permute(0, 2, 3, 1).reshape(P, s.hd)
    st["hx"][:, s.Cx:s.Cx + s.hd] = st["h"][:, :s.hd]
    st["mf"][:, :s.c_inp] = inp.permute(0, 2, 3, 1).reshape(P, s.c_inp)
    for _ in range(iters):
        disp = disp + update_v2(s, st, geo, disp)
    net_t = _nchw(st["h"], B, H, W, 0, s.hd)
    mask = .25 * m.update_block.mask(net_t.contiguous(memory_format=torch.channels_last))
    xspx = m.spx_2_gru(mask, stem_2x)
    up = ops.ctx_up(m.spx_gru(xspx), disp, 4.0).unsqueeze(1)
    return up
