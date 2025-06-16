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
import torchvision.ops as _ops

# -----------------------------------------------------------------------------
# Log noise off
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Torch‑vision NMS patched to CPU to avoid GPU serialisation cost
_orig_nms = _ops.nms

def _cpu_nms(boxes, scores, iou):
    return _orig_nms(boxes.cpu(), scores.cpu(), iou)

_ops.nms = _cpu_nms

# -----------------------------------------------------------------------------
# YOLOP classes (BDD‑10)
# Correct class order per YOLOP (see cfg/categories.txt)
YOLOP_NAMES = [
    "car", "truck", "bus", "person", "rider",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

# -----------------------------------------------------------------------------
# Inference resolution – divisible by 32 for both nets
PROC_W, PROC_H = 640, 384
SEG_W, SEG_H   = (math.ceil(v/32)*32 for v in (PROC_W, PROC_H))

# -----------------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Load HybridNets (drivable‑area mask)
hn = torch.hub.load("datvuthanh/hybridnets", "hybridnets", pretrained=True,
                    trust_repo=True).to(device).eval()
if device.type == "cuda":
    hn = hn.half()

# -----------------------------------------------------------------------------
# Load YOLOP (object detection only – keep FP32, half breaks)
yolop = torch.hub.load("hustvl/YOLOP", "yolop", pretrained=True,
                       trust_repo=True).to(device).eval()

# -----------------------------------------------------------------------------
# Add YOLOP repo root to sys.path so we can import its utils
import inspect
root_dir = Path(inspect.getfile(yolop.__class__)).resolve().parent.parent.parent  # …/hustvl_YOLOP_main
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from lib.core.general import non_max_suppression as yolop_nms

# -----------------------------------------------------------------------------
# Video I/O
VIDEO_IN  = r"C:\Users\jiami\Videos\2025-06-15_17-06-56-front.mp4"
VIDEO_OUT = "output_yolop_fix.avi"

cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open input video"
ret, sample = cap.read()
assert ret, "Cannot read first frame"
orig_H, orig_W = sample.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"XVID"), fps,
                         (orig_W, orig_H))
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# -----------------------------------------------------------------------------
# Helper – HybridNets preprocessing
_hn_mean = np.array([0.485, 0.456, 0.406], np.float32)
_hn_std  = np.array([0.229, 0.224, 0.225], np.float32)

def hn_pre(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - _hn_mean) / _hn_std
    x = x.transpose(2, 0, 1)  # CHW
    t = torch.from_numpy(x).to(device)
    return t.half() if device.type == "cuda" else t.float()

# -----------------------------------------------------------------------------
# Threads / Queues
BATCH = 8
frame_q  = queue.Queue(BATCH * 2)
result_q = queue.Queue(BATCH * 2)
_SENTINEL = object()

# -----------------------------------------------------------------------------
# Reader thread

def reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_q.put(_SENTINEL)
            return
        frame_q.put(frame)

# -----------------------------------------------------------------------------
# Batch processor

def flatten_yolop_outputs(pred):
    """Convert YOLOP multiscale outputs to (bs, N, 5+cls) tensor."""
    if not isinstance(pred, (tuple, list)):
        return pred  # already a tensor

    pieces = []
    for p in pred:
        for t in (p if isinstance(p, (list, tuple)) else [p]):
            if t.dim() == 5:          # (bs, anchors, h, w, c)
                bs, a, h, w, c = t.shape
                t = t.reshape(bs, a * h * w, c)
            elif t.dim() == 4:        # (bs, h, w, c)
                bs, h, w, c = t.shape
                t = t.reshape(bs, h * w, c)
            elif t.dim() == 3:
                pass
            else:
                continue
            pieces.append(t)
    return torch.cat(pieces, 1)


def process_batch(frames, prev_t):
    """Run both models on a mini‑batch and queue visualised frames."""
    now = time.time()
    fps_inst = 1.0 / ((now - prev_t) / len(frames))
    prev_t = now

    # Resize & convert ---------------------------------------------------
    small_bgr = [cv2.resize(f, (PROC_W, PROC_H)) for f in frames]
    small_rgb = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in small_bgr]

    # HybridNets input (with pad to 32‑multiple)
    seg_in = []
    for rgb in small_rgb:
        canvas = np.zeros((SEG_H, SEG_W, 3), np.uint8)
        canvas[:PROC_H, :PROC_W] = rgb
        seg_in.append(hn_pre(canvas))
    seg_in = torch.stack(seg_in)  # (bs,3,H,W)

    # YOLOP input (0‑1 float32)
    det_in = torch.stack([
        torch.from_numpy(b.transpose(2, 0, 1)).float().div_(255.0)
        for b in small_bgr
    ]).to(device)

    # Forward -----------------------------------------------------------
    with torch.inference_mode():
        _, _, _, _, seg = hn(seg_in)
        det_raw, _da, _ll = yolop(det_in)
        det_flat = flatten_yolop_outputs(det_raw)
        dets = yolop_nms(det_flat, conf_thres=0.7, iou_thres=0.4)

    # Post‑process -------------------------------------------------------
    masks_small = seg.argmax(1).cpu().numpy()[:, :PROC_H, :PROC_W]
    sx, sy = orig_W / PROC_W, orig_H / PROC_H
    scale = orig_H / 720.0
    fnt_scale = 0.7 * scale
    t_bg, t_fg = max(2, int(3*scale)), max(1, int(1.2*scale))

    for frame, mk_s, det in zip(frames, masks_small, dets):
        mask = cv2.resize(mk_s.astype(np.uint8), (orig_W, orig_H), cv2.INTER_NEAREST)
        overlay = frame.copy()
        overlay[mask == 2] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if det is not None and len(det):
            det = det.cpu()
            det[:, :4] *= torch.tensor([sx, sy, sx, sy])
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

        for col in [(0,0,0), (255,255,255)]:
            cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10,int(30*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        fnt_scale, col, t_bg if col==(0,0,0) else t_fg)

        result_q.put(vis)

    return prev_t

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
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
