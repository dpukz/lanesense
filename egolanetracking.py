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
import torchvision.transforms as transforms

# -------------------------------------------------------------------------
# Ensure timm==0.5.4 is installed in this environment:
#   pip install timm==0.5.4
# Then restart this script so timm 0.5.4 is loaded.

# -------------------------------------------------------------------------
# Stub out timm.models.beit.gen_relative_position_index so MiDaS hub code won't crash if it tries to import it.
try:
    import timm.models.beit as _timm_beit
except ImportError:
    import types
    _timm_beit = types.ModuleType("timm.models.beit")
    sys.modules["timm.models.beit"] = _timm_beit

if not hasattr(_timm_beit, "gen_relative_position_index"):
    def gen_relative_position_index(*args, **kwargs):
        # Dummy stub; MiDaS_small should not actually use it.
        return None
    _timm_beit.gen_relative_position_index = gen_relative_position_index

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
# Inference resolution – divisible by 32 for both nets (Do not change)
PROC_W, PROC_H = 640, 384
SEG_W, SEG_H = (math.ceil(v/32)*32 for v in (PROC_W, PROC_H))

# -------------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------------
# Depth scaling factor: multiplies the relative depth (= 1/disp) so values are larger
DEPTH_SCALE = 100.0

# -------------------------------------------------------------------------
# Kalman filter parameters for smoothing scaled relative depth:
DT = None  # will set after reading FPS
Q = np.array([[1.0, 0.0],
              [0.0, 1.0]], dtype=np.float32)
R = 10.0

# -------------------------------------------------------------------------
# Simple tracking & Kalman smoothing state:
depth_tracks = []
TRACK_X_THRESH = 50
MAX_MISSES = 5

# -------------------------------------------------------------------------
# Load HybridNets (drivable-area mask)
hn = torch.hub.load("datvuthanh/hybridnets", "hybridnets", pretrained=True,
                    trust_repo=True).to(device).eval()
if device.type == "cuda":
    hn = hn.half()

# -------------------------------------------------------------------------
# Load YOLOP (object detection only)
yolop = torch.hub.load("hustvl/YOLOP", "yolop", pretrained=True,
                       trust_repo=True).to(device).eval()
if device.type == "cuda":
    yolop = yolop.half()

# -------------------------------------------------------------------------
# Add YOLOP repo root to sys.path so we can import its utils
import inspect
root_dir = Path(inspect.getfile(yolop.__class__)).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from lib.core.general import non_max_suppression as yolop_nms

# -------------------------------------------------------------------------
# Load MiDaS_small for depth estimation (ResNet-based)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
if device.type == "cuda":
    midas = midas.half()
# Load MiDaS transforms for small model:
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # small variant

# -------------------------------------------------------------------------
# Pre-allocate normalization tensors for HybridNets
_hn_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
_hn_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# -------------------------------------------------------------------------
# HybridNets preprocessing helper (CPU fallback)
def hn_pre_cpu(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)
    t = torch.from_numpy(x).to(device)
    return t.half() if device.type == "cuda" else t.float()

# -------------------------------------------------------------------------
# Video I/O (update paths as needed)
VIDEO_IN  = r"C:\Users\Shadow\Downloads\2025-06-15_16-34-41-front.mp4"
VIDEO_OUT = "output_yolop_depth_smoothed_kalman_scaled.avi"

cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), f"Cannot open input video {VIDEO_IN}"
ret, sample = cap.read()
assert ret, "Cannot read first frame"
orig_H, orig_W = sample.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
DT = 1.0 / fps  # time step for Kalman

# Use a fast codec if possible to reduce write overhead; here XVID is kept, but you may switch to MJPG or others if supported.
writer = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (orig_W, orig_H)
)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# -------------------------------------------------------------------------
BATCH = 8
frame_q  = queue.Queue(BATCH * 2)
result_q = queue.Queue(BATCH * 2)
_SENTINEL = object()

# -------------------------------------------------------------------------
# Depth skipping: only run depth every N frames
DEPTH_SKIP = 6  # can tune higher if acceptable
_last_depth_maps = []

# -------------------------------------------------------------------------
# Pre-allocated buffer for segmentation canvas input
seg_canvas_buffer = None  # will allocate on first batch in GPU

# -------------------------------------------------------------------------
def reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_q.put(_SENTINEL)
            return
        frame_q.put(frame)

def flatten_yolop_outputs(pred):
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

def kalman_predict(state: np.ndarray, P: np.ndarray) -> (np.ndarray, np.ndarray):
    F_mat = np.array([[1.0, DT],
                      [0.0, 1.0]], dtype=np.float32)
    state_pred = F_mat @ state
    P_pred = F_mat @ P @ F_mat.T + Q
    return state_pred, P_pred

def kalman_update(state_pred: np.ndarray, P_pred: np.ndarray, meas: float) -> (np.ndarray, np.ndarray):
    y = meas - state_pred[0]
    S = P_pred[0,0] + R
    K0 = P_pred[0,0] / S
    K1 = P_pred[1,0] / S
    state_upd = state_pred.copy()
    state_upd[0] = state_pred[0] + K0 * y
    state_upd[1] = state_pred[1] + K1 * y
    P_upd = P_pred.copy()
    P_upd[0,0] = (1 - K0) * P_pred[0,0]
    P_upd[0,1] = (1 - K0) * P_pred[0,1]
    P_upd[1,0] = -K1 * P_pred[0,0] + P_pred[1,0]
    P_upd[1,1] = -K1 * P_pred[0,1] + P_pred[1,1]
    return state_upd, P_upd

def process_batch(frames, prev_t, global_frame_idx):
    global depth_tracks, _last_depth_maps, seg_canvas_buffer
    now = time.time()
    fps_inst = 1.0 / ((now - prev_t) / len(frames))
    prev_t = now

    batch_size = len(frames)
    # ----------------------------
    # Depth inference (batched), with shape-fix for MiDaS output
    run_depth = (global_frame_idx % DEPTH_SKIP == 0)
    depth_maps = []
    if run_depth:
        # Prepare CPU tensors for MiDaS
        cpu_tensors = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cpu_tensors.append(transform(img))  # each is [1, C, H_m, W_m]
        inp_batch = torch.cat(cpu_tensors, dim=0).to(device, non_blocking=True)  # [B, C, H_m, W_m]
        if device.type == "cuda" and inp_batch.dtype != torch.half:
            inp_batch = inp_batch.half()
        with torch.no_grad():
            pred = midas(inp_batch)  # could be [B, H_o, W_o] or [B,1,H_o,W_o]
            # --- Ensure shape is [B,1,H_o,W_o] ---
            if pred.ndim == 3:
                # [B, H, W] -> [B,1,H,W]
                pred = pred.unsqueeze(1)
            elif pred.ndim == 4 and pred.shape[1] != 1:
                # e.g. [B, H, W, 1] or channels>1
                # If last dim==1, permute; else take first channel
                if pred.shape[-1] == 1:
                    pred = pred.permute(0, 3, 1, 2)
                else:
                    pred = pred[:, :1, ...]
            # Now pred is [B,1,H_o,W_o]
            pred_resized = F.interpolate(pred, size=(PROC_H, PROC_W), mode='bilinear', align_corners=False)
            disp_batch = pred_resized[:, 0, :, :]  # [B, PROC_H, PROC_W]
            depth_rel = 1.0 / (disp_batch + 1e-6)
            depth_scaled = depth_rel * DEPTH_SCALE
            depth_np_batch = depth_scaled.cpu().numpy().astype(np.float32)  # [B, PROC_H, PROC_W]
            depth_maps = [depth_np_batch[i] for i in range(depth_np_batch.shape[0])]
        _last_depth_maps = depth_maps.copy()
    else:
        if _last_depth_maps and len(_last_depth_maps) == batch_size:
            depth_maps = _last_depth_maps
        else:
            depth_maps = [np.zeros((PROC_H, PROC_W), np.float32) for _ in frames]
    # ----------------------------
    # Preprocessing for HybridNets and YOLOP
    use_gpu_resize = False
    if device.type == "cuda":
        try:
            np_batch = np.stack(frames, axis=0)  # [B, H, W, 3]
            ft = torch.from_numpy(np_batch).permute(0,3,1,2).to(device, non_blocking=True).float()
            rgb = ft[:, [2,1,0], :, :]
            small_rgb = F.interpolate(rgb, size=(PROC_H, PROC_W), mode='bilinear', align_corners=False)
            small_bgr = small_rgb[:, [2,1,0], :, :]
            use_gpu_resize = True
        except Exception:
            use_gpu_resize = False

    if use_gpu_resize:
        # HybridNets input
        rgb_norm = small_rgb.div_(255.0)
        hn_in = (rgb_norm - _hn_mean) / _hn_std  # [B,3,PROC_H,PROC_W]
        # Pre-allocate/reuse seg_canvas_buffer
        if seg_canvas_buffer is None or seg_canvas_buffer.shape[0] < BATCH:
            dtype = hn_in.dtype
            seg_canvas_buffer = torch.zeros((BATCH, 3, SEG_H, SEG_W), device=device, dtype=dtype)
        seg_canvas_buffer[:batch_size, :, :PROC_H, :PROC_W] = hn_in
        seg_in = seg_canvas_buffer[:batch_size]
        # YOLOP input
        det_in = small_bgr.div_(255.0).to(device, non_blocking=True)
    else:
        seg_list = []
        det_tensors = []
        for f in frames:
            small_b = cv2.resize(f, (PROC_W, PROC_H))
            small_rgb = cv2.cvtColor(small_b, cv2.COLOR_BGR2RGB)
            canvas = np.zeros((SEG_H, SEG_W, 3), np.uint8)
            canvas[:PROC_H, :PROC_W] = small_rgb
            seg_list.append(hn_pre_cpu(canvas))
            t = torch.from_numpy(small_b.transpose(2,0,1)).float().div_(255.0)
            det_tensors.append(t)
        seg_in = torch.stack(seg_list).to(device, non_blocking=True)
        det_in = torch.stack(det_tensors).to(device, non_blocking=True)

    # ----------------------------
    # Inference
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                _, _, _, _, seg = hn(seg_in)
                det_raw, _, _ = yolop(det_in)
        else:
            _, _, _, _, seg = hn(seg_in)
            det_raw, _, _ = yolop(det_in)

        det_flat = flatten_yolop_outputs(det_raw)
        # GPU NMS if possible
        if det_flat is not None and det_flat.device.type == 'cuda':
            try:
                dets = yolop_nms(det_flat, conf_thres=0.7, iou_thres=0.4)
            except Exception:
                dets = yolop_nms(det_flat.cpu(), conf_thres=0.7, iou_thres=0.4)
        else:
            if det_flat is not None and det_flat.device.type != 'cpu':
                det_flat_cpu = det_flat.cpu()
            else:
                det_flat_cpu = det_flat
            dets = yolop_nms(det_flat_cpu, conf_thres=0.7, iou_thres=0.4)

    # Postprocess & visualization
    masks_small = seg.argmax(1).cpu().numpy()[:, :PROC_H, :PROC_W]
    sx, sy = orig_W / PROC_W, orig_H / PROC_H
    scale = orig_H / 720.0
    fnt_scale = 0.7 * scale
    t_bg, t_fg = max(2, int(3*scale)), max(1, int(1.2*scale))
    scale_tensor = torch.tensor([sx, sy, sx, sy], device='cpu')

    matched_tracks = set()

    for idx, det in enumerate(dets):
        frame = frames[idx]
        mask_small = masks_small[idx]
        for i in range(len(mask_small)):
            # find the middle column and then go immediately left and right to find the innermost lane (value == 2) and then take their mean, that's the center of the ego lane
            rowlen = mask_small[i].size
            basel = rowlen // 2
            baser = basel + 1
            while basel > 0 and mask_small[i][basel] != 2:
                basel -= 1
            while baser < rowlen and mask_small[i][baser] != 2:
                baser += 1
            if basel > 0 and baser < rowlen:
                midpoint = (basel + baser) // 2
                for j in range(-4, 4):
                    mask_small[i][midpoint + j] = 3
        mask = cv2.resize(mask_small.astype(np.uint8), (orig_W, orig_H), cv2.INTER_CUBIC)
        overlay = frame.copy()
        overlay[mask == 2] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        depth_map = depth_maps[idx]  # [PROC_H, PROC_W]

        if det is not None and len(det):
            if det.device.type == 'cuda':
                det_cpu = det.cpu()
            else:
                det_cpu = det.clone()
            det_cpu[:, :4] *= scale_tensor
            for x1, y1, x2, y2, conf, cls in det_cpu.numpy():
                if conf < 0.45:
                    continue
                xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))

                # Sample median depth
                x_center = (xi1 + xi2) / 2.0
                y_bottom = yi2
                x_proc = int(x_center / sx)
                y_proc = int(y_bottom / sy)
                x_proc = max(0, min(x_proc, PROC_W - 1))
                y_proc = max(0, min(y_proc, PROC_H - 1))
                x0 = max(0, x_proc - 2)
                x1p = min(PROC_W - 1, x_proc + 2)
                y0 = max(0, y_proc - 2)
                y1p = min(PROC_H - 1, y_proc + 2)
                patch = depth_map[y0:y1p+1, x0:x1p+1]
                if patch.size > 0:
                    depth_val = float(np.median(patch))
                else:
                    depth_val = float(depth_map[y_proc, x_proc])

                # Kalman smoothing per track
                matched = None
                for ti, tr in enumerate(depth_tracks):
                    if tr['class'] == int(cls) and abs(tr['x'] - x_center) < TRACK_X_THRESH:
                        matched = (ti, tr)
                        break

                if matched is not None:
                    ti, tr = matched
                    try:
                        state_pred, P_pred = kalman_predict(tr['state'], tr['P'])
                    except Exception:
                        state_pred = np.array([depth_val, 0.0], dtype=np.float32)
                        P_pred = np.eye(2, dtype=np.float32)
                    if not (math.isnan(depth_val) or np.isinf(depth_val)):
                        try:
                            state_upd, P_upd = kalman_update(state_pred, P_pred, depth_val)
                        except Exception:
                            state_upd = np.array([depth_val, 0.0], dtype=np.float32)
                            P_upd = np.eye(2, dtype=np.float32)
                    else:
                        state_upd, P_upd = state_pred, P_pred
                    tr['state'] = state_upd
                    tr['P'] = P_upd
                    tr['x'] = x_center
                    tr['miss'] = 0
                    depth_disp = float(state_upd[0])
                    matched_tracks.add(ti)
                else:
                    if math.isnan(depth_val) or np.isinf(depth_val):
                        continue
                    state0 = np.array([depth_val, 0.0], dtype=np.float32)
                    P0 = np.eye(2, dtype=np.float32)
                    new_tr = {
                        'class': int(cls),
                        'x': x_center,
                        'state': state0,
                        'P': P0,
                        'miss': 0
                    }
                    depth_tracks.append(new_tr)
                    ti_new = len(depth_tracks) - 1
                    matched_tracks.add(ti_new)
                    depth_disp = float(state0[0])

                # Draw box & labels
                label = YOLOP_NAMES[int(cls)] if cls < len(YOLOP_NAMES) else str(int(cls))
                conf_text = f"{label} c={conf:.2f}"
                depth_text = f"d={depth_disp:.2f}"

                ts = fnt_scale
                (sz1_w, sz1_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, ts, t_fg)
                (sz2_w, sz2_h), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, ts, t_fg)
                max_w = max(sz1_w, sz2_w)
                x_text = xi1
                if x_text + max_w > orig_W:
                    x_text = max(0, orig_W - max_w - 1)
                y_line1 = max(yi1 - 5, sz1_h + sz2_h + 10)
                y_line2 = y_line1 - (sz1_h + 2)

                cv2.rectangle(vis, (xi1, yi1), (xi2, yi2), (255, 0, 255), max(1, int(2*scale)))
                for col in [(0,0,0), (255,0,255)]:
                    cv2.putText(vis, conf_text, (x_text, y_line1),
                                cv2.FONT_HERSHEY_SIMPLEX, ts, col, t_bg if col==(0,0,0) else t_fg)
                for col in [(0,0,0), (255,0,255)]:
                    cv2.putText(vis, depth_text, (x_text, y_line2),
                                cv2.FONT_HERSHEY_SIMPLEX, ts, col, t_bg if col==(0,0,0) else t_fg)

        # FPS overlay
        for col in [(0,0,0), (255,255,255)]:
            cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10,int(30*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        fnt_scale, col, t_bg if col==(0,0,0) else t_fg)

        result_q.put(vis)

    # Age and prune tracks
    for i, tr in enumerate(depth_tracks):
        if i not in matched_tracks:
            tr['miss'] += 1
    depth_tracks[:] = [tr for tr in depth_tracks if tr['miss'] <= MAX_MISSES]

    return prev_t

def worker():
    prev = time.time()
    batch = []
    global_frame_idx = 0
    while True:
        f = frame_q.get()
        if f is _SENTINEL:
            if batch:
                prev = process_batch(batch, prev, global_frame_idx)
            result_q.put(_SENTINEL)
            return
        batch.append(f)
        if len(batch) >= BATCH:
            prev = process_batch(batch, prev, global_frame_idx)
            global_frame_idx += len(batch)
            batch = []

threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

print("running – press q to quit")
last_disp = time.time()
FRAME_INTERVAL = 1.0 / fps
while True:
    frame = result_q.get()
    if frame is _SENTINEL:
        break
    now = time.time()
    elapsed = now - last_disp
    if elapsed < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - elapsed)
    last_disp = time.time()

    writer.write(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
