"""General (any size, no per-size compile) inference patches for Fast-FoundationStereo.

    from core.inference_patch import apply; apply(model)

1. geometry lookup: grid_sample with cuDNN disabled. cuDNN's grid sampler rejects N = B*H*W at 1/4 resolution of
   large inputs (Midd-F: CUDNN_STATUS_NOT_SUPPORTED) -- the same fix FoundationStereo got upstream (8063753).
   Native kernel, identical math.
2. group-wise correlation volume: upstream materializes the expanded, normalized fp32 left/right volumes
   (C x D x H x W each, 2 x 7 GB at Midd-F) and is @torch.compile'd (per-shape compilation). Here: per-pixel
   normalization once, then per row block one fp32 GEMM against the W-flipped right features; disparity d of pixel
   w is the strided diagonal element (w, W-1-w+d) (entries w < d are zero, as upstream). Same math.
3. concat volume: upstream function without @torch.compile.
"""
import torch
import torch.nn.functional as F

from core import geometry, submodule
import core.foundation_stereo as fsmod


def bilinear_sampler_nocudnn(img, coords, mode='bilinear', mask=False, low_memory=False, use1d=False):
    H, W = img.shape[-2:]
    coords[..., 0] = 2 * coords[..., 0] / (W - 1) - 1
    with torch.backends.cudnn.flags(enabled=False):
        return F.grid_sample(img, coords, align_corners=True)


def build_gwc_volume_gemm(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True, rows=16):
    dtype = refimg_fea.dtype
    B, C, H, W = refimg_fea.shape
    G, D = num_groups, maxdisp
    vol = torch.empty((B, G, D, H, W), device=refimg_fea.device, dtype=dtype)
    valid = torch.arange(W, device=refimg_fea.device).view(1, W) >= torch.arange(D, device=refimg_fea.device).view(D, 1)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        h = r1 - r0
        f1 = refimg_fea[:, :, r0:r1].float().reshape(B, G, C // G, h, W)
        f2 = targetimg_fea[:, :, r0:r1].float().reshape(B, G, C // G, h, W)
        if normalize:
            # upstream rounds the normalized features to the AMP dtype before the product
            f1 = F.normalize(f1, dim=2).to(dtype).float()
            f2 = F.normalize(f2, dim=2).to(dtype).float()
        f1 = f1.permute(0, 1, 3, 4, 2).reshape(B * G * h, W, C // G)
        f2 = f2.flip(-1).permute(0, 1, 3, 2, 4).reshape(B * G * h, C // G, W)
        M = torch.bmm(f1, f2)
        V = M.as_strided((B * G * h, D, W), (W * W, 1, W - 1), M.storage_offset() + W - 1)
        vol[:, :, :, r0:r1] = torch.where(valid, V, 0.0).view(B, G, h, D, W).permute(0, 1, 3, 2, 4)
    return vol


def _exporting():
    return torch.compiler.is_exporting() or torch.compiler.is_compiling() or torch.onnx.is_in_onnx_export()


def build_gwc_volume_shifted(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
    """== build_gwc_volume_optimized_pytorch1: disparity d pairs pixel w with right pixel w-d, zero for w < d."""
    dtype = refimg_fea.dtype
    B, C, H, W = refimg_fea.shape
    G, D = num_groups, maxdisp
    f1 = refimg_fea.reshape(B, G, C // G, H, W)
    f2 = targetimg_fea.reshape(B, G, C // G, H, W)
    if normalize:   # upstream rounds the normalized features to the AMP dtype before the product
        f1 = F.normalize(f1.float(), dim=2).to(dtype)
        f2 = F.normalize(f2.float(), dim=2).to(dtype)
    f2 = F.pad(f2, (D - 1, 0))
    return torch.stack([(f1 * f2[..., D - 1 - d:D - 1 - d + W]).sum(2) for d in range(D)], 2)


def build_concat_volume_shifted(refimg_fea, targetimg_fea, maxdisp):
    """== build_concat_volume_optimized_pytorch1 (ref broadcast over disparities, right shifted with zero fill)."""
    B, C, H, W = refimg_fea.shape
    D = maxdisp
    tp = F.pad(targetimg_fea, (D - 1, 0))
    tv = torch.stack([tp[..., D - 1 - d:D - 1 - d + W] for d in range(D)], 2)
    return torch.cat((refimg_fea.unsqueeze(2).expand(B, C, D, H, W), tv), dim=1)


def _orig(f):
    return getattr(f, "_torchdynamo_orig_callable", None) or getattr(f, "__wrapped__", f)


def apply(model):
    geometry.bilinear_sampler = bilinear_sampler_nocudnn
    fsmod.build_gwc_volume_optimized_pytorch1 = build_gwc_volume_gemm
    fsmod.build_concat_volume_optimized_pytorch1 = _orig(submodule.build_concat_volume_optimized_pytorch1)
    return model


# ---------------------------------------------------------------------------------------------- fast path (pure fp16)
# prepare_fast(model); forward_fast(model, L, R, iters=8)
#  - pure fp16 weights / activations, channels_last(_3d), no autocast re-casts; disparity state fp32
#  - layout-preserving InstanceNorm (fp32 statistics); features one image at a time
#  - geometry volume kept fp16 in NDHWC (no fp32 permuted copy), level 1 by D-pair pooling, all-pairs correlation fp16,
#    gather lookup (== grid_sample align_corners=True, zero padding), no host syncs; convc1 input channels permuted once
import torch.nn as nn


def _mf(x):
    return torch.channels_last if x.dim() == 4 else torch.channels_last_3d


def _instance_norm_nhwc(self, x):
    dims = tuple(range(2, x.dim()))
    var, mean = torch.var_mean(x.float(), dim=dims, keepdim=True, unbiased=False)
    y = (x.float() - mean) * torch.rsqrt(var + self.eps)
    if self.affine:
        shp = (1, -1) + (1,) * (x.dim() - 2)
        y = y * self.weight.float().view(shp) + self.bias.float().view(shp)
    return y.to(x.dtype)


_BN = (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
_CONV = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def _fold(conv, bn):
    s = bn.weight.detach().float() * torch.rsqrt(bn.running_var.float() + bn.eps)
    t = bn.bias.detach().float() - bn.running_mean.float() * s
    w = conv.weight.data.float()
    shp = (1, -1) + (1,) * (w.dim() - 2) if isinstance(conv, (nn.ConvTranspose2d, nn.ConvTranspose3d)) else (-1,) + (1,) * (w.dim() - 1)
    conv.weight.data = w * s.view(shp)
    b = conv.bias.data.float() if conv.bias is not None else torch.zeros_like(s)
    conv.bias = nn.Parameter(b * s + t)


def fold_batchnorms(model):
    """Eval BatchNorm directly after a conv folded into the conv (fp32, before the fp16 cast)."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Sequential, nn.ModuleList)):
            ch = list(mod._modules.items())
            for (k0, a), (k1, b) in zip(ch, ch[1:]):
                if isinstance(a, _CONV) and isinstance(b, _BN):
                    _fold(a, b); mod._modules[k1] = nn.Identity(); n += 1
        for cname, bname in (("conv", "bn"), ("conv1", "bn1"), ("conv2", "bn2")):
            c, b = getattr(mod, cname, None), getattr(mod, bname, None)
            if isinstance(c, _CONV) and isinstance(b, _BN):
                _fold(c, b); setattr(mod, bname, nn.Identity()); n += 1
    return n


def prepare_fast(model):
    apply(model)
    model.eval()
    model._nfold = fold_batchnorms(model)
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)) and not m.track_running_stats:
            m.forward = _instance_norm_nhwc.__get__(m)
    C, K, L = model.volume_dim, 2 * model.args.corr_radius + 1, model.args.corr_levels
    perm, base = [], 0
    for _ in range(L):
        perm += [base + c * K + k for k in range(K) for c in range(C)]
        base += C * K
        perm += [base + k for k in range(K)]
        base += K
    cc1 = model.update_block.encoder.convc1
    cc1.weight.data = cc1.weight.data[:, perm].contiguous()
    model.half()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)) and m.weight.dim() in (4, 5):
            m.weight.data = m.weight.data.contiguous(memory_format=_mf(m.weight.data))
    model._dxf = torch.arange(-model.args.corr_radius, model.args.corr_radius + 1, device=cc1.weight.device).float()
    return model


class GeoFast:
    def __init__(self, fl, fr, comb, levels, dx):
        B, C, D, H, W = comb.shape
        self.B, self.C, self.H, self.W, self.levels, self.dx = B, C, H, W, levels, dx
        g = comb.permute(0, 2, 3, 4, 1)
        self.geo = [g.contiguous()]
        for _ in range(levels - 1):
            g = self.geo[-1]
            Dn = g.shape[1] // 2
            self.geo.append((0.5 * (g[:, 0:2 * Dn:2] + g[:, 1:2 * Dn:2])).contiguous())
        f1 = F.normalize(fl.float(), dim=1).to(comb.dtype).permute(0, 2, 3, 1).reshape(B * H, W, -1)
        f2 = F.normalize(fr.float(), dim=1).to(comb.dtype).permute(0, 2, 3, 1).reshape(B * H, W, -1)
        self.corr = [torch.bmm(f1, f2.transpose(1, 2)).view(B, H, W, W)]
        for _ in range(levels - 1):
            c = self.corr[-1]
            Wn = c.shape[-1] // 2
            self.corr.append((0.5 * (c[..., 0:2 * Wn:2] + c[..., 1:2 * Wn:2])).contiguous())
        self.wpos = torch.arange(W, device=comb.device, dtype=torch.float32).view(1, 1, W, 1)
        hw = torch.arange(H * W, device=comb.device).view(1, H, W, 1)
        self.b = torch.arange(B, device=comb.device).view(B, 1, 1, 1)
        self.hw = hw

    @staticmethod
    def _lerp_idx(base, stride, pos, n):
        x0 = torch.floor(pos)
        f = pos - x0
        i0 = x0.long()
        i1 = i0 + 1
        w0 = (1 - f) * ((i0 >= 0) & (i0 <= n - 1)).to(pos.dtype)
        w1 = f * ((i1 >= 0) & (i1 <= n - 1)).to(pos.dtype)
        return base + i0.clamp(0, n - 1) * stride, base + i1.clamp(0, n - 1) * stride, w0, w1

    def __call__(self, disp):
        B, H, W, C = self.B, self.H, self.W, self.C
        d = disp.view(B, H, W, 1)
        outs = []
        for i in range(self.levels):
            s = 2 ** i
            g = self.geo[i]
            Di = g.shape[1]
            dt = g.dtype
            r0, r1, w0, w1 = self._lerp_idx(self.b * Di * H * W + self.hw, H * W, d / s + self.dx, Di)
            gf = g.view(-1, C)
            outs.append((gf[r0] * w0.unsqueeze(-1).to(dt) + gf[r1] * w1.unsqueeze(-1).to(dt)).view(B, H, W, -1))
            c = self.corr[i]
            W2 = c.shape[-1]
            r0, r1, w0, w1 = self._lerp_idx((self.b * H * W + self.hw) * W2, 1, self.wpos / s - d / s + self.dx, W2)
            cf = c.view(-1)
            outs.append(cf[r0] * w0.to(dt) + cf[r1] * w1.to(dt))
        return torch.cat(outs, dim=-1).permute(0, 3, 1, 2)


@torch.no_grad()
def forward_fast(m, image1, image2, iters=8):
    i1, i2 = fsmod.normalize_image(image1), fsmod.normalize_image(image2)
    cl = lambda x: x.half().contiguous(memory_format=_mf(x))
    fl = m.feature(cl(i1))
    fr = m.feature(cl(i2))
    stem_2x = m.stem_2(cl(i1))
    D = m.args.max_disp // 4
    if _exporting():   # same values from shifted views (no as_strided / Tensor.unfold / per-row slice assignment)
        gwc = build_gwc_volume_shifted(fl[0], fr[0], D, m.cv_group, normalize=m.args.normalize)
        concat = build_concat_volume_shifted(m.proj_cmb(fl[0]), m.proj_cmb(fr[0]), D)
    else:
        gwc = build_gwc_volume_gemm(fl[0], fr[0], D, m.cv_group, normalize=m.args.normalize)
        concat = fsmod.build_concat_volume_optimized_pytorch1(m.proj_cmb(fl[0]), m.proj_cmb(fr[0]), maxdisp=D)
    comb = torch.cat([gwc, concat], dim=1).contiguous(memory_format=torch.channels_last_3d)
    del gwc, concat
    comb = m.corr_stem(comb)
    comb = m.corr_feature_att(comb, fl[0])
    comb = m.cost_agg(comb, fl)
    prob = F.softmax(m.classifier(comb).squeeze(1).float(), dim=1)
    disp = submodule.disparity_regression(prob, D)
    del prob
    cnet = list(m.cnet(fl[0], fl[1], fl[2]))
    net = [torch.tanh(x[0]) for x in cnet]
    inp = [torch.relu(x[1]) for x in cnet]
    inp = [m.cam(x) * x for x in inp]
    att = [m.sam(x) for x in inp]
    geo = GeoFast(fl[0], fr[0], comb.contiguous(memory_format=torch.channels_last_3d), m.args.corr_levels, m._dxf)
    del comb, fl, fr, cnet
    for _ in range(iters):
        corr = geo(disp)
        net, mask, delta = m.update_block(net, inp, corr, disp.half(), att)
        disp = disp + delta.float()
    with torch.autocast("cuda", enabled=False):
        xspx = m.spx_2_gru(mask, stem_2x)
        spx = F.softmax(m.spx_gru(xspx).float(), 1)
        up = submodule.context_upsample(disp * 4.0, spx).unsqueeze(1)
    return up.float()
