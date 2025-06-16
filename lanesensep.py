import math
import logging
import cv2
import time
import threading
import queue
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as _ops

# -------------------------------------------------------------------------
# Log noise off
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# -------------------------------------------------------------------------
# YOLOP classes (BDD-10)
YOLOP_NAMES = [
    "car", "truck", "bus", "person", "rider",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

# -------------------------------------------------------------------------
# Inference resolution – divisible by 32 for both nets
PROC_W, PROC_H = 640, 384
SEG_W, SEG_H = (math.ceil(v/32)*32 for v in (PROC_W, PROC_H))

# -------------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------------
# Load HybridNets (drivable-area mask)
hn = torch.hub.load("datvuthanh/hybridnets", "hybridnets", pretrained=True,
                    trust_repo=True).to(device).eval()
if device.type == "cuda":
    hn = hn.half()

# -------------------------------------------------------------------------
# Load YOLOP (object detection only – keep FP32 by default)
yolop = torch.hub.load("hustvl/YOLOP", "yolop", pretrained=True,
                       trust_repo=True).to(device).eval()
# If YOLOP safely supports half precision, optionally: yolop = yolop.half()

# -------------------------------------------------------------------------
# Add YOLOP repo root to sys.path so we can import its utils
import inspect
root_dir = Path(inspect.getfile(yolop.__class__)).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from lib.core.general import non_max_suppression as yolop_nms

# -------------------------------------------------------------------------
# Video I/O
VIDEO_IN  = r"C:\Users\jiami\Videos\2025-06-15_17-06-56-front.mp4"
VIDEO_OUT = "output_yolop_fix.avi"

cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open input video"
ret, sample = cap.read()
assert ret, "Cannot read first frame"
orig_H, orig_W = sample.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

writer = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (orig_W, orig_H)
)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# -------------------------------------------------------------------------
# Helper – HybridNets preprocessing (for CPU fallback)
_hn_mean = np.array([0.485, 0.456, 0.406], np.float32)
_hn_std  = np.array([0.229, 0.224, 0.225], np.float32)
def hn_pre_cpu(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - _hn_mean) / _hn_std
    x = x.transpose(2, 0, 1)  # CHW
    t = torch.from_numpy(x).to(device)
    return t.half() if device.type == "cuda" else t.float()

# -------------------------------------------------------------------------
# Threads / Queues
BATCH = 8
frame_q  = queue.Queue(BATCH * 2)
result_q = queue.Queue(BATCH * 2)
_SENTINEL = object()

# -------------------------------------------------------------------------
# Reader thread
def reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_q.put(_SENTINEL)
            return
        frame_q.put(frame)

# -------------------------------------------------------------------------
# Batch processor helpers
def flatten_yolop_outputs(pred):
    """Convert YOLOP multiscale outputs to (bs, N, 5+cls) tensor."""
    if not isinstance(pred, (tuple, list)):
        return pred
    pieces = []
    for p in pred:
        for t in (p if isinstance(p, (list, tuple)) else [p]):
            if t.dim() == 5:
                bs, a, h, w, c = t.shape
                t = t.reshape(bs, a * h * w, c)
            elif t.dim() == 4:
                bs, h, w, c = t.shape
                t = t.reshape(bs, h * w, c)
            elif t.dim() == 3:
                pass
            else:
                continue
            pieces.append(t)
    return torch.cat(pieces, 1)

def process_batch(frames, prev_t):
    now = time.time()
    fps_inst = 1.0 / ((now - prev_t) / len(frames))
    prev_t = now

    # Attempt GPU-based resize + color conversion:
    use_gpu_resize = False
    if device.type == "cuda":
        try:
            # Stack frames into a tensor: shape (B, H, W, 3), dtype uint8
            np_batch = np.stack(frames, axis=0)  # shape (B, orig_H, orig_W, 3)
            # Convert to tensor, permute to (B,3,H,W), float
            frame_tensor = torch.from_numpy(np_batch).permute(0,3,1,2).to(device, non_blocking=True).float()
            # BGR -> RGB:
            rgb_tensor = frame_tensor[:, [2,1,0], :, :]  # still 0..255
            # Resize to (PROC_H, PROC_W) using interpolate
            small_rgb_tensor = F.interpolate(rgb_tensor.unsqueeze(0) if False else rgb_tensor,
                                             size=(PROC_H, PROC_W),
                                             mode='bilinear',
                                             align_corners=False)
            # Now small_rgb_tensor: shape (B,3,PROC_H,PROC_W)
            # Also get small_bgr for detection input
            small_bgr_tensor = small_rgb_tensor[:, [2,1,0], :, :]
            use_gpu_resize = True
        except Exception as e:
            # Fallback to CPU resizing if anything fails
            use_gpu_resize = False

    if use_gpu_resize:
        # Prepare HybridNets input on GPU
        # Normalize small_rgb_tensor from [0,255] to [0,1], then standardize
        rgb_norm = small_rgb_tensor.div_(255.0)
        # Apply mean/std: broadcast over batch and spatial dims
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        hn_in = (rgb_norm - mean) / std  # shape (B,3,PROC_H,PROC_W)
        # Pad to SEG_H, SEG_W: create zeros and copy
        B = hn_in.shape[0]
        seg_canvas = torch.zeros((B, 3, SEG_H, SEG_W), device=device, dtype=hn_in.dtype)
        seg_canvas[:, :, :PROC_H, :PROC_W] = hn_in
        seg_in = seg_canvas  # ready for HybridNets

        # Prepare YOLOP input on GPU: small_bgr_tensor normalized [0,1]
        det_in = small_bgr_tensor.div_(255.0).to(device)
    else:
        # CPU fallback: use cv2.resize and hn_pre_cpu
        small_bgr = [cv2.resize(f, (PROC_W, PROC_H)) for f in frames]
        small_rgb = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in small_bgr]
        # HybridNets input CPU -> to device
        seg_list = []
        for rgb in small_rgb:
            canvas = np.zeros((SEG_H, SEG_W, 3), np.uint8)
            canvas[:PROC_H, :PROC_W] = rgb
            seg_list.append(hn_pre_cpu(canvas))
        seg_in = torch.stack(seg_list)
        # YOLOP input
        det_in = torch.stack([
            torch.from_numpy(b.transpose(2, 0, 1)).float().div_(255.0)
            for b in small_bgr
        ]).to(device)

    # Forward with mixed precision on CUDA if available
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                _, _, _, _, seg = hn(seg_in)
                det_raw, _da, _ll = yolop(det_in)
        else:
            _, _, _, _, seg = hn(seg_in)
            det_raw, _da, _ll = yolop(det_in)

        det_flat = flatten_yolop_outputs(det_raw)
        # NMS with fallback
        try:
            dets = yolop_nms(det_flat, conf_thres=0.7, iou_thres=0.4)
        except NotImplementedError:
            det_flat_cpu = det_flat.cpu()
            dets = yolop_nms(det_flat_cpu, conf_thres=0.7, iou_thres=0.4)

    # Post-process segmentation mask
    masks_small = seg.argmax(1).cpu().numpy()[:, :PROC_H, :PROC_W]
    sx, sy = orig_W / PROC_W, orig_H / PROC_H
    scale = orig_H / 720.0
    fnt_scale = 0.7 * scale
    t_bg, t_fg = max(2, int(3*scale)), max(1, int(1.2*scale))
    scale_tensor = torch.tensor([sx, sy, sx, sy], device='cpu')

    for idx, det in enumerate(dets):
        # For overlay, we need the original frame
        frame = frames[idx]
        # Segmentation overlay via CPU numpy mask
        mk_s = masks_small[idx]
        mask = cv2.resize(mk_s.astype(np.uint8), (orig_W, orig_H), cv2.INTER_NEAREST)
        overlay = frame.copy()
        overlay[mask == 2] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Detection boxes
        if det is not None and len(det):
            # Move to CPU and clone to avoid in-place errors
            if det.device.type != 'cpu':
                det = det.cpu()
            det = det.clone()
            det[:, :4] *= scale_tensor
            for x1, y1, x2, y2, conf, cls in det.numpy():
                if conf < 0.45:
                    continue
                xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
                label = YOLOP_NAMES[int(cls)] if cls < len(YOLOP_NAMES) else str(int(cls))
                txt = f"{label}:{conf:.2f}"
                cv2.rectangle(vis, (xi1, yi1), (xi2, yi2), (255, 0, 255), max(1, int(2*scale)))
                for col in [(0,0,0), (255,0,255)]:
                    cv2.putText(vis, txt, (xi1, max(yi1-5,0)), cv2.FONT_HERSHEY_SIMPLEX,
                                fnt_scale, col, t_bg if col==(0,0,0) else t_fg)

        # FPS overlay
        for col in [(0,0,0), (255,255,255)]:
            cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10,int(30*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        fnt_scale, col, t_bg if col==(0,0,0) else t_fg)

        result_q.put(vis)

    return prev_t

# -------------------------------------------------------------------------
# Worker thread
def worker():
    prev = time.time()
    batch = []
    while True:
        f = frame_q.get()
        if f is _SENTINEL:
            if batch:
                process_batch(batch, prev)
            result_q.put(_SENTINEL)
            return
        batch.append(f)
        if len(batch) >= BATCH:
            prev = process_batch(batch, prev)
            batch = []

# -------------------------------------------------------------------------
# Spin up threads
threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

print("running – press q to quit")
while True:
    frame = result_q.get()
    if frame is _SENTINEL:
        break
    writer.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
