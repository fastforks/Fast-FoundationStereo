"""Upstream-friendly v2 (plain PyTorch, ONNX/TensorRT-exportable) Fast-FS path: the exportable subset of the max v2 changes
(cuda-kernels branch) on top of core/inference_patch (friendly v1):
  * GRU: every activation zero-padded to a multiple of 8 channels (weights padded once, exact), z|r convs merged, mask head only
    after the last iteration, one concat per input instead of several
  * cost-volume path: volume_dim (28) channels zero-padded to 32 (Conv3d / ConvTranspose3d / FeatureAtt weights padded once;
    the disparity attention runs on the 28 real channels)
  * classifier Conv3d(28 -> 1, 3x3x3) as one 2-D conv with the 3 disparity taps as output channels + shifted sum
  * channel attention with plain mean / amax reductions
No custom kernels, no in-place tricks that exporters reject.

    from core.inference_friendly2 import prepare_friendly2, forward_friendly2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import core.foundation_stereo as fsmod
from core import submodule
from core import inference_patch as ip


def p8(c):
    return (c + 7) // 8 * 8


def _wb(mod):
    w = mod.weight.detach().float()
    if w.dim() == 2:
        w = w[:, :, None, None]
    return w, (mod.bias.detach().float() if mod.bias is not None else None)


class PConv(nn.Module):
    """zero-padded conv2d (srcs: (module | (w, b), out0, n_out, out_dst); in_segs: (in0, n_in, in_dst)) + activation."""
    ACTS = {0: lambda x: x, 1: torch.relu, 5: F.gelu, 7: torch.sigmoid, 8: torch.tanh}

    def __init__(self, srcs, in_segs, cin, kout, padding, act=0):
        super().__init__()
        first = srcs[0][0]
        w0, _ = first if isinstance(first, tuple) else _wb(first)
        W = torch.zeros(kout, cin, *w0.shape[2:], device=w0.device)
        B = torch.zeros(kout, device=w0.device)
        for mod, o0, no, od in srcs:
            w, b = mod if isinstance(mod, tuple) else _wb(mod)
            for i0, ni, idst in in_segs:
                W[od:od + no, idst:idst + ni] = w[o0:o0 + no, i0:i0 + ni]
            if b is not None:
                B[od:od + no] = b[o0:o0 + no]
        self.weight = nn.Parameter(W.half().contiguous(memory_format=torch.channels_last), requires_grad=False)
        self.bias = nn.Parameter(B.half(), requires_grad=False)
        self.padding = tuple(padding)
        self.act = act

    def forward(self, x):
        return self.ACTS[self.act](F.conv2d(x, self.weight, self.bias, 1, self.padding))


def _pad_conv_(mod, cin, cout):
    w = mod.weight.data.float()
    tr = isinstance(mod, (nn.ConvTranspose2d, nn.ConvTranspose3d))
    W = w.new_zeros(((cin, cout) if tr else (cout, cin)) + tuple(w.shape[2:]))
    W[:w.shape[0], :w.shape[1]] = w
    mf = torch.channels_last_3d if W.dim() == 5 else torch.channels_last
    mod.weight.data = W.half().contiguous(memory_format=mf)
    if mod.bias is not None:
        b = mod.bias.data.float(); B = b.new_zeros(cout); B[:b.numel()] = b
        mod.bias = nn.Parameter(B.half(), requires_grad=False)
    mod.in_channels, mod.out_channels = cin, cout


def pad_cost_path(m, Cv):
    C = m.volume_dim
    for root in (m.corr_stem, m.corr_feature_att, m.cost_agg):
        for mod in root.modules():
            if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)) and mod.groups == 1:
                ci = Cv if mod.in_channels == C else mod.in_channels
                co = Cv if mod.out_channels == C else mod.out_channels
                if (ci, co) != (mod.in_channels, mod.out_channels):
                    _pad_conv_(mod, ci, co)
            if isinstance(mod, submodule.FeatureAtt) and mod.feat_att[1].out_channels == C:
                _pad_conv_(mod.feat_att[1], mod.feat_att[1].in_channels, Cv)
            if isinstance(mod, submodule.CostVolumeDisparityAttention) and not hasattr(mod, "_fr2"):
                mod._fr2 = True
                fw = mod.forward

                def att_fwd(cv, window_size=(-1, -1), fw=fw):
                    y = fw(cv[:, :C], window_size)
                    return F.pad(y, (0, 0, 0, 0, 0, 0, 0, Cv - C))
                mod.forward = att_fwd


def prepare_friendly2(m):
    ip.prepare_fast(m)            # friendly v1: BN folded, fp16, NHWC, convc1 input in the GeoFast order
    ub = m.update_block
    enc, gru, dh = ub.encoder, ub.gru04, ub.disp_head
    C, K, L = m.volume_dim, 2 * m.args.corr_radius + 1, m.args.corr_levels
    hd = gru.small_gru.convz.out_channels; Hd = p8(hd)
    cx = gru.conv0[0].out_channels; Cx = p8(cx)
    c_inp = m.cam.fc[0].in_channels
    c_mo = enc.conv.out_channels
    ncorr = L * K * (C + 1)
    s = nn.Module()
    s.C, s.K, s.Hd, s.hd, s.Cx, s.cx, s.c_inp, s.c_mo, s.ncorr, s.Ccorr = C, K, Hd, hd, Cx, cx, c_inp, c_mo, ncorr, p8(ncorr)
    s.convc1 = PConv([(enc.convc1, 0, enc.convc1.out_channels, 0)], [(0, ncorr, 0)], s.Ccorr, p8(enc.convc1.out_channels), (0, 0), 1)
    c2, d2 = enc.convc2.out_channels, enc.convd2.out_channels
    s.convc2 = PConv([(enc.convc2, 0, c2, 0)], [(0, enc.convc2.in_channels, 0)], p8(enc.convc2.in_channels), p8(c2), (1, 1), 1)
    s.convd1 = PConv([(enc.convd1, 0, enc.convd1.out_channels, 0)], [(0, 1, 0)], 8, p8(enc.convd1.out_channels), (3, 3), 1)
    s.convd2 = PConv([(enc.convd2, 0, d2, 0)], [(0, enc.convd2.in_channels, 0)], p8(enc.convd2.in_channels), p8(d2), (1, 1), 1)
    s.conv = PConv([(enc.conv, 0, c_mo, 0)], [(0, c2, 0), (c2, d2, p8(c2))], p8(c2) + p8(d2), p8(c_mo), (1, 1), 1)
    s.Cmf = p8(c_inp + c_mo + 1)
    assert c_inp % 8 == 0 and c_mo % 8 == 0
    s.conv0 = PConv([(gru.conv0[0], 0, cx, 0)], [(0, c_inp + c_mo + 1, 0)], s.Cmf, Cx, (1, 1), 1)
    c1o = gru.conv1[0].out_channels
    s.conv1 = PConv([(gru.conv1[0], 0, c1o, 0)], [(0, cx, 0), (cx, hd, Cx)], Cx + Hd, p8(c1o), (1, 1), 1)
    zr_in = [(0, gru.small_gru.convz.in_channels, 0)]
    s.zr_s = PConv([(gru.small_gru.convz, 0, hd, 0), (gru.small_gru.convr, 0, hd, Hd)], zr_in, p8(c1o), 2 * Hd, (0, 0), 7)
    s.zr_l = PConv([(gru.large_gru.convz, 0, hd, 0), (gru.large_gru.convr, 0, hd, Hd)], zr_in, p8(c1o), 2 * Hd, (1, 1), 7)
    q_in = [(0, hd, 0), (hd, cx, Hd)]
    s.q_s = PConv([(gru.small_gru.convq, 0, hd, 0)], q_in, Hd + Cx, Hd, (0, 0), 8)
    s.q_l = PConv([(gru.large_gru.convq, 0, hd, 0)], q_in, Hd + Cx, Hd, (1, 1), 8)
    seq = dh.conv
    c36 = seq[0].out_channels; Cdh = p8(c36)
    s.dh0 = PConv([(seq[0], 0, c36, 0)], [(0, hd, 0)], Hd, Cdh, (1, 1), 1)
    s.enx = nn.ModuleList()
    for blk in (seq[2], seq[3]):
        e = blk.pwconv1.out_features
        dw = nn.Conv2d(Cdh, Cdh, blk.dwconv.kernel_size, padding=blk.dwconv.padding, groups=Cdh).to(blk.dwconv.weight.device)
        dw.weight.data.zero_(); dw.bias.data.zero_()
        dw.weight.data[:c36] = blk.dwconv.weight.detach().float(); dw.bias.data[:c36] = blk.dwconv.bias.detach().float()
        dw.half(); dw.weight.data = dw.weight.data.contiguous(memory_format=torch.channels_last)
        g = torch.zeros(Cdh, device=dw.weight.device); g[:c36] = blk.gamma.detach().float() if blk.gamma is not None else 1.0
        mm = nn.Module()
        mm.dw = dw
        mm.pw1 = PConv([(blk.pwconv1, 0, e, 0)], [(0, c36, 0)], Cdh, p8(e), (0, 0), 5)
        mm.pw2 = PConv([(blk.pwconv2, 0, c36, 0)], [(0, e, 0)], p8(e), Cdh, (0, 0), 0)
        mm.register_buffer("gamma", g.half().view(1, Cdh, 1, 1))
        s.enx.append(mm)
    s.dh_out = PConv([(seq[4], 0, 1, 0)], [(0, c36, 0)], Cdh, 8, (1, 1), 0)
    cls = m.classifier.layers[0] if hasattr(m.classifier, "layers") else m.classifier[0]
    wc = cls.weight.detach().float()
    s.Ccls = p8(C)
    s.cls = PConv([((wc[:, :, kd], None), 0, 1, kd) for kd in range(3)], [(0, C, 0)], s.Ccls, 8, (1, 1), 0)
    s.register_buffer("cls_b", cls.bias.detach().float().view(1) if cls.bias is not None else torch.zeros(1, device=wc.device))
    pad_cost_path(m, s.Ccls)
    m.fr2 = s
    return m


def classifier_fr2(s, comb):
    """[B, Cv, D, H, W] -> soft-argmax disparity [B, 1, H, W] fp32 (Conv3d(C, 1, 3, pad 1) + softmax + regression)."""
    B, C, D, H, W = comb.shape
    x = comb.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    o = s.cls(x).float().reshape(B, D, 8, H, W)
    z = o.new_zeros(B, 1, H, W)
    cost = o[:, :, 1] + s.cls_b.view(1, 1, 1, 1) + torch.cat([z, o[:, :-1, 0]], 1) + torch.cat([o[:, 1:, 2], z], 1)
    prob = F.softmax(cost, dim=1)
    return submodule.disparity_regression(prob, D)


def cam_fr2(cam, x):
    return torch.sigmoid(cam.fc(x.mean((2, 3), keepdim=True)) + cam.fc(x.amax((2, 3), keepdim=True)))


@torch.no_grad()
def forward_friendly2(m, image1, image2, iters=8):
    s = m.fr2
    i1, i2 = fsmod.normalize_image(image1), fsmod.normalize_image(image2)
    cl = lambda x: x.half().contiguous(memory_format=ip._mf(x))
    fl = m.feature(cl(i1))
    fr = m.feature(cl(i2))
    stem_2x = m.stem_2(cl(i1))
    D = m.args.max_disp // 4
    if ip._exporting():
        gwc = ip.build_gwc_volume_shifted(fl[0], fr[0], D, m.cv_group, normalize=m.args.normalize)
        concat = ip.build_concat_volume_shifted(m.proj_cmb(fl[0]), m.proj_cmb(fr[0]), D)
    else:
        gwc = ip.build_gwc_volume_gemm(fl[0], fr[0], D, m.cv_group, normalize=m.args.normalize)
        concat = fsmod.build_concat_volume_optimized_pytorch1(m.proj_cmb(fl[0]), m.proj_cmb(fr[0]), maxdisp=D)
    comb = torch.cat([gwc, concat], dim=1).contiguous(memory_format=torch.channels_last_3d)
    del gwc, concat
    comb = m.corr_stem(comb)
    comb = m.corr_feature_att(comb, fl[0])
    comb = m.cost_agg(comb, fl)
    disp = classifier_fr2(s, comb)
    cnet = list(m.cnet(fl[0], fl[1], fl[2]))
    net = torch.tanh(cnet[0][0])
    inp = torch.relu(cnet[0][1])
    inp = cam_fr2(m.cam, inp) * inp
    att = m.sam(inp)
    geo = ip.GeoFast(fl[0], fr[0], comb[:, :s.C].contiguous(memory_format=torch.channels_last_3d), m.args.corr_levels, m._dxf)
    del comb, fl, fr, cnet
    B, _, H, W = net.shape
    h = F.pad(net, (0, 0, 0, 0, 0, s.Hd - s.hd))
    zpad = disp.new_zeros(B, 7, H, W).half()
    zc = disp.new_zeros(B, s.Ccorr - s.ncorr, H, W).half()
    for _ in range(iters):
        corr = geo(disp)                                       # [B, 522, H, W] (GeoFast order)
        corr = torch.cat([corr, zc], 1).contiguous(memory_format=torch.channels_last)
        d8 = torch.cat([disp.half(), zpad], 1)
        c2 = s.convc2(s.convc1(corr))
        dd = s.convd2(s.convd1(d8))
        mo = s.conv(torch.cat([c2, dd], 1))
        x = s.conv0(torch.cat([inp, mo, d8], 1))
        hx2 = s.conv1(torch.cat([x, h], 1))
        zrs, zrl = s.zr_s(hx2), s.zr_l(hx2)
        zs, rs = zrs[:, :s.Hd], zrs[:, s.Hd:]
        zl, rl = zrl[:, :s.Hd], zrl[:, s.Hd:]
        qs = s.q_s(torch.cat([rs * h, x], 1))
        ql = s.q_l(torch.cat([rl * h, x], 1))
        h = att * ((1 - zs) * h + zs * qs) + (1 - att) * ((1 - zl) * h + zl * ql)
        t = s.dh0(h)
        for b in s.enx:
            t = t + b.gamma * b.pw2(b.pw1(b.dw(t)))
        disp = disp + s.dh_out(t)[:, :1].float()
    mask = .25 * m.update_block.mask(h[:, :s.hd])
    with torch.autocast("cuda", enabled=False):
        xspx = m.spx_2_gru(mask, stem_2x)
        spx = F.softmax(m.spx_gru(xspx).float(), 1)
        up = submodule.context_upsample(disp * 4.0, spx).unsqueeze(1)
    return up.float()
