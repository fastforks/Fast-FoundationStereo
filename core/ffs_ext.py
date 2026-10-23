"""Prebuilt CUDA extension of the Fast-FS max path (core/csrc/ffs_fused.cu): CUTLASS fused conv2d (+bias +residual +act,
fp16 accumulate, NHWC, output into channel slices) + Fast-FS kernels (geo lookup, GRU gates).
Compiled once per source hash into core/prebuilt/ffs_<hash>/ (SASS sm_80/86/89/90/120 + PTX, built against torch 2.14 cu130) and
loaded with torch.ops.load_library: no compiler needed at run time, nothing compiled or tuned per input size. Ops: torch.ops.ffs.*
Without a matching prebuilt it is built on first use when CUDA_HOME is set (nvcc matching torch's CUDA, CUTLASS v4.7.1 headers in
CUTLASS_DIR):  CUDA_HOME=... CUTLASS_DIR=... python -m core.ffs_ext
"""
import hashlib
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc", "ffs_fused.cu")
_loaded = [False]


def _paths():
    tag = hashlib.sha1(open(SRC, "rb").read()).hexdigest()[:12]
    bdir = os.path.join(HERE, "prebuilt", f"ffs_{tag}")
    return tag, bdir, os.path.join(bdir, f"ffs_fused_{tag}.so")


def load():
    if _loaded[0]:
        return
    tag, bdir, so = _paths()
    if not os.path.exists(so):
        if not os.environ.get("CUDA_HOME"):
            raise RuntimeError(f"no prebuilt {so}: set CUDA_HOME and CUTLASS_DIR to build it")
        build()
    torch.ops.load_library(so)
    _loaded[0] = True


def build():
    tag, bdir, so = _paths()
    cutlass = os.environ["CUTLASS_DIR"]
    os.makedirs(bdir, exist_ok=True)
    from torch.utils.cpp_extension import load as _load
    libdir = os.path.join(HERE, "prebuilt", "lib")
    os.makedirs(libdir, exist_ok=True)
    cudart = os.path.join(os.environ.get("CUDA_HOME", ""), "lib", "libcudart.so.13")
    if os.path.exists(cudart) and not os.path.exists(os.path.join(libdir, "libcudart.so")):
        os.symlink(cudart, os.path.join(libdir, "libcudart.so"))
    arch = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (80, 86, 89, 90, 120)] + ["-gencode=arch=compute_80,code=compute_80"]
    _load(name=f"ffs_fused_{tag}", sources=[SRC], build_directory=bdir, verbose=True, is_python_module=False,
          extra_include_paths=[os.path.join(cutlass, "include")], extra_ldflags=[f"-L{libdir}"],
          extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__",
                             "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", *arch])
    for f in os.listdir(bdir):   # keep only the .so (build intermediates are not shipped)
        if not f.endswith(".so"):
            p = os.path.join(bdir, f)
            if os.path.isfile(p):
                os.remove(p)
    return so


if __name__ == "__main__":
    print(build())
