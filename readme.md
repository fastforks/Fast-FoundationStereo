# Fast-FoundationStereo: faster inference fork

> **You are on `cuda-kernels`.** Research fork of [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (base `476f424`, 2026-09-11; `main` mirrors upstream). Two branches:
> - [`portable`](https://github.com/fastforks/Fast-FoundationStereo/tree/portable): plain PyTorch (`core/inference_friendly2.py`; the simpler `core/inference_patch.py` is about as fast in eager mode), exports to ONNX / TensorRT, no build step, no triton / torch.compile.
> - [`cuda-kernels`](https://github.com/fastforks/Fast-FoundationStereo/tree/cuda-kernels): `portable` plus custom CUDA kernels (`core/inference_max2.py`, torch extension `core/csrc/ffs_fused.cu`; prebuilt for sm_80/86/89/90/120 + PTX against torch 2.14 cu130, built on first use from `CUDA_HOME` + `CUTLASS_DIR` when no prebuilt matches).
>
> Licence, as upstream: code and weights NVIDIA licence (non-commercial / research only, see [LICENSE.txt](LICENSE.txt))

```python
from core.inference_friendly2 import prepare_friendly2, forward_friendly2   # portable
prepare_friendly2(model); disp = forward_friendly2(model, L, R, iters=8)
from core.inference_max2 import prepare_max2, forward_max2                  # cuda-kernels
prepare_max2(model); disp = forward_max2(model, L, R, iters=8)
```

Fast-FoundationStereo, released weights `23-36-37` (upstream default), valid_iters 8, max_disp 192, fp16 autocast (upstream, cost volume `pytorch1` via `torch.compile`) / fp16 (forks), pad /32; RTX 5060 Ti 16 GB, torch 2.14.1+cu130, TensorRT 11.4, 2026-10-23. Median of 10 runs after 3 warm-ups; memory = torch peak allocated / reserved (TensorRT: engine device memory). EPE on 6 ETH3D and 3 Middlebury-H training pairs (short check, not a full eval).

| version | KITTI ms | Midd-H ms | Midd-F ms | speed-up at H | peak MB at H (alloc / reserved) | peak MB at F (alloc / reserved) | startup / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|---|---|
| upstream | 108.9 | 407.2 | not measured | 1.00x | 2879 / 3490 | not measured | 3.3 s (+6.2 s first call, compile) / 5.1 s | 0.324 | 0.314 |
| **portable** | 99.1 | 327.9 | not measured | 1.24x | 2070 / 2518 | not measured | 3.4 s / 0 ms | 0.323 | 0.313 |
| portable → TensorRT | 48.4 | 143.5 | not measured | 2.84x | ctx 1085 | not measured | 1.7 s / none | not measured | not measured |
| **cuda-kernels** | 49.2 | 144.0 | not measured | 2.83x | 1829 / 2350 | not measured | 3.5 s / 30 ms | 0.323 | 0.315 |

---

# Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching

This is the official implementation of our paper accepted to CVPR 2026

[[Website]](https://nvlabs.github.io/Fast-FoundationStereo/) [[Paper]](https://arxiv.org/abs/2512.11130) [[Video]](https://www.youtube.com/watch?v=2BUYZojCzXE)

Authors: [Bowen Wen](https://wenbowen123.github.io/), [Shaurya Dewan](https://www.linkedin.com/in/shaurya-dewan-1b07231a2), [Stan Birchfield](https://research.nvidia.com/person/stan-birchfield)


# Abstract
Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications. Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning. To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate. We employ a divide-and-conquer acceleration strategy with three components: (1) knowledge distillation to compress the hybrid backbone into a single efficient student; (2) blockwise neural architecture search for automatically discovering optimal cost filtering designs under latency budgets, reducing search complexity exponentially; and (3) structured pruning for eliminating redundancy in the iterative refinement module. Furthermore, we introduce an automatic pseudo-labeling pipeline used to curate 1.4M in-the-wild stereo pairs to supplement synthetic training data and facilitate knowledge distillation. The resulting model can run over 10× faster than FoundationStereo while closely matching its zero-shot accuracy, thus establishing a new state-of-the-art among real-time methods.

 [NOTE] This model is designed for real-time applications. For offline computation for the best accuracy, please checkout our earlier work [FoundationStereo](https://github.com/NVlabs/FoundationStereo).



<p align="center">
  <img src="assets/intro.jpg" width="100%"/>
</p>

<td align="center">
  <img src="assets/intro_c.webp" width="60%"/>
</td>
<td align="center">
  <img src="assets/bp2_vs_runtime.jpg" width="60%"/>
</td>


# Environment setup
- Option 1: Docker
```bash
docker build --network host -t ffs -f docker/dockerfile .
bash docker/run_container.sh
```

- Option 2: pip
```bash
conda create -n ffs python=3.12 && conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```


# Weights and Trade-off
**Commercial use:** Download [C-Fast-FoundationStereo](https://huggingface.co/nvidia/c-fast-foundationstereo) under the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). For commercially licensed training and inference code, see [NVIDIA TAO Toolkit](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/depth_estimation/fast_foundation_stereo.html). 

**Research checkpoints:** Download from [Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and put under the folder `weights/` (e.g. `./weights/23-36-37`). Below table compares the differences among some representative models of varying sizes from our trained family. They are sorted from slowest to fastest, with accuracy descending, where runtime is profiled on GPU 3090, image size 640x480.

To trade-off speed and accuracy, there are two options:
1) Try with different checkpoints.
2) Tune the config flags (see explanations in the "Run demo" section below).

| Checkpoint     | valid_iters | Runtime-Pytorch (ms) | Runtime-TRT (ms) | Peak Memory (MB) |
|---------------|-------------|-------------|-----------------|-----------------|
| `23-36-37`    | 8           | 49.4        | 23.4            | 653             |
| `23-36-37`    | 4           | 41.1        | 18.4            | 653             |
| `20-26-39`    | 8           | 43.6        | 19.4            | 651             |
| `20-26-39`    | 4           | 37.5        | 16.4            | 651             |
| `20-30-48`    | 8           | 38.4        | 16.6            | 646             |
| `20-30-48`    | 4           | 29.3        | 14.0            | 646             |

# Run demo
```
python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100
```
| Flag                        | Meaning                                                                |
|-----------------------------|------------------------------------------------------------------------|
| `--model_dir`               | Path to the trained weights/model file                                 |
| `--left_file`               | Path to the left image file                                            |
| `--right_file`              | Path to the right image file                                           |
| `--intrinsic_file`          | Path to the camera intrinsic matrix and baseline file                  |
| `--out_dir`                 | Output directory for saving results                                    |
| `--remove_invisible`        | Whether to ignore non-overlapping region's depth (0: no, 1: yes)      |
| `--denoise_cloud`           | Whether to apply denoising to the point cloud (0: no, 1: yes)          |
| `--scale`                   | Image scaling factor                                                   |
| `--get_pc`                  | Obtain point cloud output (0: no, 1: yes)                              |
| `--valid_iters`             | Number of refinement updates during forward pass                       |
| `--max_disp`                | Maximum disparity for volume encoding, 192 should be enough, unless you need to sense very near objects (e.g. <0.1m). Increasing it runs slower and uses more memory. |
| `--zfar`                    | Maximum depth to include in point cloud                                |

Refer to `scripts/run_demo.py` for comprehensive list of flags.

**Tips:**
- The input left and right images should be rectified and undistorted, which means there should not be fisheye kind of lens distortion and the epipolar lines are horizontal between the left/right images. If you obtain images from stereo cameras such as Zed, they usually have handled this for you.
- Do not swap left and right image. The left image should really be obtained from the left-side camera (objects will appear righter in the image).
- We recommend to use PNG files with no lossy compression
- Our method works best on stereo RGB images. However, we have also tested it on monochrome or IR stereo images (e.g. from RealSense D4XX series) and it works well too.
- To get point cloud for your own data, you need to specify the intrinsics. In the intrinsic file in args, 1st line is the flattened 1x9 intrinsic matrix, 2nd line is the baseline (distance) between the left and right camera, unit in meters.
- The model performs better for image width size <1000. You can run with smaller scale, e.g. `--scale 0.5` to downsize input image, then upsize the output depth to your need with nearest neighbor interpolation.
- For faster inference, you can reduce the input image resolution by e.g. `--scale 0.5`, and reduce refine iterations by e.g. `--valid_iters 4`.
- Note that the 1st time running is slower due to compilation, use a while loop after warm up for live running.

Expect to see results like below:
- Disparity/Depth:
  <p align="center">
    <img src="assets/disp_vis.png" alt="Disparity Visualization" width="100%">
  </p>

- Point cloud:
  <p align="center">
    <img src="assets/pcl_vis.png" alt="Point Cloud Visualization" width="100%">
  </p>


# ONNX/TRT
For TRT, we recommend first setup env in docker.

## Single ONNX

Export the full model as a single ONNX file. This replaces the Triton GWC kernel with ONNX-compatible ops so no intermediate engine split is needed.

```bash
python scripts/make_single_onnx.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --save_path output/ --height 480 --width 640 --valid_iters 8 --max_disp 192
```

| Flag              | Meaning                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `--model_dir`     | Path to the trained weights/model file                                   |
| `--save_path`     | Directory to save the ONNX model and config                              |
| `--height`        | Input image height, must be divisible by 32. Reduce for faster speed.    |
| `--width`         | Input image width, must be divisible by 32. Reduce for faster speed.     |
| `--valid_iters`   | Number of refinement updates during forward pass, reduce for faster speed, but may drop quality |
| `--max_disp`      | Maximum disparity for volume encoding, 192 should be enough, unless you need to sense very near objects (e.g. <0.1m). Increasing it runs slower and uses more memory. |
| `--onnx_name`     | Base name for the saved ONNX file (default: `fast_foundationstereo`)     |

Then convert to a single TRT engine:
```bash
trtexec --onnx=output/fast_foundationstereo.onnx --saveEngine=output/fast_foundationstereo.engine --fp16
```

To run inference with the single ONNX or TRT engine:
```bash
python scripts/run_demo_single_trt.py --model_dir output/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output_demo/ --get_pc 1 --remove_invisible 0 --denoise_cloud 1 --zfar 100
```

The script auto-detects `.engine` or `.onnx` files in `--model_dir`. To use a specific file, pass `--model_file` directly.

| Flag                  | Meaning                                                                  |
|-----------------------|--------------------------------------------------------------------------|
| `--model_dir`         | Directory containing the .onnx/.engine file and its .yaml config         |
| `--model_file`        | Explicit path to .onnx or .engine file (overrides auto-search)           |
| `--left_file`         | Path to the left image file                                              |
| `--right_file`        | Path to the right image file                                             |
| `--intrinsic_file`    | Path to the camera intrinsic matrix and baseline file                    |
| `--out_dir`           | Output directory for saving results                                      |
| `--remove_invisible`  | Whether to ignore non-overlapping region's depth (0: no, 1: yes)         |
| `--denoise_cloud`     | Whether to apply denoising to the point cloud (0: no, 1: yes)            |
| `--get_pc`            | Obtain point cloud output (0: no, 1: yes)                                |
| `--zfar`              | Maximum depth (m) to include in point cloud                              |

**Note:** The single ONNX model expects **pre-normalized** float32 inputs (ImageNet normalization stripped). The inference script handles this automatically. If integrating into your own pipeline, apply normalization beforehand:
```
normalized = (pixel - mean) / std
mean = [123.675, 116.28, 103.53]
std  = [ 58.395, 57.12, 57.375]
```

## Two-stage ONNX

The original export splits the model into two ONNX files around the Triton GWC kernel, which runs as an intermediate step between the two TRT engines.

```bash
python scripts/make_onnx.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --save_path output/ --height 448 --width 640 --valid_iters 8 --max_disp 192
```

| Flag              | Meaning                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `--model_dir`     | Path to the trained weights/model file                                   |
| `--save_path`     | Directory to save ONNX outputs and zip file                             |
| `--height`        | Input image height, better to be divisible by 32. Reduce image size can increase speed.                                          |
| `--width`         | Input image width,  better to be divisible by 32. Reduce image size can increase speed.                                     |
| `--valid_iters`   | Number of updates during forward pass, reduce it for faster speed, but may drop quality                                    |
| `--max_disp`      | Maximum disparity for volume encoding, 192 should be enough, unless you need to sense very near objects (e.g. <0.1m). Increasing it runs slower and uses more memory.                                    |

Refer to `scripts/make_onnx.py` for a comprehensive list of available flags.

Then convert from ONNX to TRT:
```bash
trtexec --onnx=output/feature_runner.onnx --saveEngine=output/feature_runner.engine --fp16  --useCudaGraph
trtexec --onnx=output/post_runner.onnx --saveEngine=output/post_runner.engine --fp16  --useCudaGraph
```

To use the two-stage TRT for inference:
```bash
python scripts/run_demo_tensorrt.py --onnx_dir output/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --get_pc 1 --zfar 100
```

# Internet-Scale Pseudo-Labeling
Real-world data offers greater diversity and realism than synthetic data. However, obtaining real stereo images with ground-truth metric depth annotation is notoriously difficult. To address this challenge, we propose an automatic data curation pipeline to generate pseudo-labels on internet-scale stereo images from [Stereo4D](https://stereo4d.github.io/) dataset. **Top:** Pseudo-labeling pipeline on in-the-wild internet stereo data. **Bottom:** Visualization of our generated pseudo-labels.

<p align="center">
  <img src="assets/stereo4d.jpg" width="50%">
</p>

Below are visualizations of the intermediate results in our pseudo-labeling process. In the rightmost column, green checkmark or red cross denotes whether samples are kept for training or not, based on the percentage of positive pixels in the consistency mask. Our data curation process can automatically discover failures on noisy internet data such as images containing subtitle (bottom), mosaic (2nd last row) and overly challenging samples that are unsuitable for training (top). The final pseudo-labels can also correct erroneous predictions from FoundationStereo on sky regions (5th row).

<p align="center">
  <img src="assets/stereo4d_labeling.jpg" width="100%">
</p>


The dataset is available at HuggingFace: https://huggingface.co/datasets/nvidia/ffs_stereo4d

# Citation
```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
```

# Contact
Please contact [Bowen Wen](https://wenbowen123.github.io/) (bowenw@nvidia.com) for questions and commercial inquiries.

# Acknowledgement
We would like to thank Xutong Ren, Karsten Patzwaldt, Yonggan Fu, Saurav Muralidharan, Han Cai, Pavlo Molchanov, Yu Wang, Varun Praveen, Joseph Aribido and Jun Gao for their insightful early discussions for this project. We would also like to thank NVIDIA Isaac and TAO teams for their engineering support and valuable discussions. Thanks to the authors of [FoundationStereo](https://github.com/NVlabs/FoundationStereo), [Selective-IGEV](https://github.com/Windsrain/Selective-Stereo), [Stereo4D](https://github.com/Stereo4d/stereo4d-code) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for their code release. Finally, thanks to CVPR reviewers and AC for their appreciation of this work and constructive feedback.