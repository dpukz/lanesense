***HybridNets × YOLOv5 Lane‑Aware Object Detection Pipeline***

Real‑time multi‑task perception: semantic lane segmentation, drivable‑area masking, and object detection fused into a single, GPU‑accelerated video processor.

**Introduction**

This repository contains a single‑file prototype (lanesense.py) that combines the strengths of two state‑of‑the‑art architectures: HybridNets for drivable‑area and lane‑line segmentation, and YOLOv5‑Nano for lightweight object detection—into a unified, real‑time video processing pipeline. The script down‑samples each frame to a work resolution (640×360 px) for inference, applies FP16 acceleration, and then re‑projects high‑confidence results back onto the original resolution for clear overlays.

**Features**

Multi‑Task Perception· Drivable‑area + lane segmentation via HybridNets· Object detection via YOLOv5n

Real‑Time through Smart Down‑ScalingWork‑resolution frames (configurable) are padded to multiples of 32 to satisfy HybridNets’ backbone, slashing compute cost without sacrificing overlay fidelity.

Batch‑Oriented ThreadingSeparate reader and worker threads feed a bounded queue, keeping the GPU saturated while ensuring the CPU never becomes the bottleneck.

FP16 & cuDNN BenchAutomatic device selection (cuda > mps > cpu) and half‑precision inference maximize throughput on modern GPUs and Apple Silicon.

Pure‑Python, Zero CUDA KernelsA tiny NMS shim forces TorchVision’s NMS onto CPU, avoiding GPU context stalls when mixing half‑precision tensors and float32 ops.

Instant Visual FeedbackLive cv2.imshow() display plus XVID recording (output_downscaled_fix.avi) at source FPS.

**Requirements**

Python ≥ 3.9

PyTorch ≥ 2.1 with matching CUDA / MPS build

TorchVision ≥ 0.16

OpenCV‑Python ≥ 4.9

Ultralytics ≥ 8.2 (for YOLOv5 API)

NumPy ≥ 1.25

Tip (Windows + GPU): install the official CUDA toolkit before pip install torch==...+cuXX to avoid missing DLL errors.

**Installation**

1. Clone Repo

2. Create & activate venv

3. Install deps

4. (optional) Install YOLOv5n weights once to avoid first-run latency

**Quick Start** 

python hybridnet.py

A preview window appears; press Q to terminate early. An annotated video will be sent saved in AVI format using the source FPS and resolution. If program freezes, force quit python in task mngr.

**CLI Arguments**

--Input    : Path to input video or webcam index (Required)

--output   : Output video file name

--proc-size: WxH for inference down-scaling. Must be <= source size.

--batch    : Number of frames per HybridNets batch (GPU VRAM trade-off).

--conf     : YOLO object-confidence threshold

--iou      : YOLO NMS IoU threshold.

**Workflow** 

1. Frame Grab — a dedicated reader thread pulls frames from cv2.VideoCapture and enqueues them.

2. Down‑Scale & Padding — frames are resized to the work resolution and padded to a 32‑pixel grid for HybridNets.

3. HybridNets Inference — semantic masks (lane, drivable area, background) are predicted in FP16.

4. YOLOv5 Inference — object detection runs on the same down‑scaled frames; boxes are later up‑scaled.

5. Post‑Processing — lane pixels are reprojected onto the original frame; boxes and labels are rendered with confidence scores.

6. Batched FPS Computation — instantaneous FPS is averaged per batch for smoother read‑outs.

7. Display & Write — annotated frames are both shown in a resizable window and written to disk.

**Acknowledgements**

Dat Vũ‑Thanh for releasing HybridNets weights.

Ultralytics for the lightweight YOLOv5 implementation & permissive license.

PyTorch team for seamless mixed‑precision APIs.
