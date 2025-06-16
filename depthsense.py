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
import torchvision.transforms as transforms

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
# Simple smoothing state for depth tracks, with adaptive smoothing
depth_tracks = []  # list of dict: {'class': int, 'x': float, 'depth': float, 'miss': int}
ALPHA_OLD_APPROACH = 0.2
ALPHA_OLD_AWAY     = 0.7
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
# Load Monodepth2 for depth estimation, with fallback to local clone
def load_monodepth2_model(model_name="mono_640x192"):
    try:
        m = torch.hub.load("nianticlabs/monodepth2", model_name).to(device).eval()
        if device.type == "cuda":
            m = m.half()
        print(f"Loaded Monodepth2 '{model_name}' via torch.hub")
        return m, None
    except Exception as e:
        print(f"torch.hub.load failed: {e}. Using local clone.")
        MONO_ROOT = Path(r"C:\Users\jiami\PathPilotDemo\monodepth2")
        if not MONO_ROOT.exists():
            raise FileNotFoundError(f"Local Monodepth2 repo not found at {MONO_ROOT}")
        sys.path.append(str(MONO_ROOT))
        from networks import ResnetEncoder, DepthDecoder
        model_folder = MONO_ROOT / "models" / model_name
        encoder_path = model_folder / "encoder.pth"
        decoder_path = model_folder / "depth.pth"
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(f"Weights not found at {encoder_path} or {decoder_path}")
        encoder = ResnetEncoder(18, False).to(device)
        loaded = torch.load(str(encoder_path), map_location=device)
        filtered = {k: v for k, v in loaded.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered)
        encoder.eval()
        decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4)).to(device)
        loaded = torch.load(str(decoder_path), map_location=device)
        decoder.load_state_dict(loaded)
        decoder.eval()
        if device.type == "cuda":
            encoder = encoder.half()
            decoder = decoder.half()
        print(f"Loaded Monodepth2 '{model_name}' via local clone")
        return (encoder, decoder), model_name

depth_res = load_monodepth2_model("mono_640x192")

# -------------------------------------------------------------------------
# Pre-allocate normalization tensors for HybridNets
_hn_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
_hn_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# -------------------------------------------------------------------------
# Preprocess for Monodepth2 using cv2.resize directly (no PIL)
def preprocess_depth(frame_bgr, model_info):
    # Determine target w,h
    if model_info is None:
        fw, fh = 640, 192
    else:
        parts = model_info.split("_")
        if len(parts) >= 2 and "x" in parts[1]:
            w_h = parts[1].split("x"); fw, fh = int(w_h[0]), int(w_h[1])
        else:
            fw, fh = 640, 192
    # Resize with cv2
    small = cv2.resize(frame_bgr, (fw, fh), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device).float() / 255.0
    if device.type == "cuda":
        tensor = tensor.half()
    # Normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # We can apply Normalize on CPU or GPU: use functional
    tensor = normalize(tensor.squeeze(0)).unsqueeze(0).to(device)
    return tensor, (fw, fh), small.shape[:2]

# -------------------------------------------------------------------------
# Video I/O (update paths)
VIDEO_IN  = r"C:\Users\jiami\Downloads\yt1z.net - 3-Hour Drive on California Freeway Dash Cam Tours (360p).mp4"
VIDEO_OUT = "output_yolop_depth_smoothed.avi"

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
# HybridNets preprocessing helper
def hn_pre_cpu(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)
    t = torch.from_numpy(x).to(device)
    return t.half() if device.type == "cuda" else t.float()

# -------------------------------------------------------------------------
BATCH = 8
frame_q  = queue.Queue(BATCH * 2)
result_q = queue.Queue(BATCH * 2)
_SENTINEL = object()

# -------------------------------------------------------------------------
# Depth skipping: only run depth every N frames
DEPTH_SKIP = 3  # run depth inference every 3 frames
# Maintain a buffer of last depth maps to reuse
_last_depth_maps = []
_last_depth_frame_indices = []

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

def process_batch(frames, prev_t, global_frame_idx):
    global depth_tracks, _last_depth_maps, _last_depth_frame_indices
    now = time.time()
    fps_inst = 1.0 / ((now - prev_t) / len(frames))
    prev_t = now

    batch_size = len(frames)
    # Depth preprocessing & inference, with skip
    depth_maps = []
    run_depth = (global_frame_idx % DEPTH_SKIP == 0)
    if run_depth:
        depth_tensors = []
        for frame in frames:
            inp, _, _ = preprocess_depth(frame, depth_res[1])
            depth_tensors.append(inp)
        depth_batch = torch.cat(depth_tensors, dim=0)
        # Depth inference
        with torch.inference_mode():
            if depth_res[1] is None:
                dm = depth_res[0]
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        disp = dm(depth_batch)
                else:
                    disp = dm(depth_batch)
            else:
                encoder, decoder = depth_res[0]
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        feats = encoder(depth_batch)
                        out = decoder(feats)
                        disp = out.get(("disp", 0), out.get("disp_0"))
                else:
                    feats = encoder(depth_batch)
                    out = decoder(feats)
                    disp = out.get(("disp", 0), out.get("disp_0"))
        depth_batch = 1.0 / (disp + 1e-6)
        depth_maps = []
        for i in range(depth_batch.shape[0]):
            d = depth_batch[i:i+1]
            d_resized = F.interpolate(d, size=(PROC_H, PROC_W), mode="bilinear", align_corners=False)
            depth_maps.append(d_resized.squeeze().cpu().numpy())
        # store for reuse
        _last_depth_maps = depth_maps.copy()
        _last_depth_frame_indices = [global_frame_idx + i for i in range(len(frames))]
    else:
        # reuse last depth maps if available; else dummy zeros
        if _last_depth_maps and len(_last_depth_maps) == batch_size:
            depth_maps = _last_depth_maps
        else:
            depth_maps = [np.zeros((PROC_H, PROC_W), np.float32) for _ in frames]

    # HybridNets/YOLOP preprocessing: attempt GPU resizing always if CUDA available
    use_gpu_resize = False
    if device.type == "cuda":
        try:
            np_batch = np.stack(frames, axis=0)
            ft = torch.from_numpy(np_batch).permute(0,3,1,2).to(device, non_blocking=True).float()
            # if half precision desired: convert after resizing
            rgb = ft[:, [2,1,0], :, :]
            small_rgb = F.interpolate(rgb, size=(PROC_H, PROC_W), mode='bilinear', align_corners=False)
            small_bgr = small_rgb[:, [2,1,0], :, :]
            use_gpu_resize = True
        except Exception:
            use_gpu_resize = False

    if use_gpu_resize:
        # Normalize for HybridNets
        rgb_norm = small_rgb.div_(255.0)
        hn_in = (rgb_norm - _hn_mean) / _hn_std
        B = hn_in.shape[0]
        seg_canvas = torch.zeros((B, 3, SEG_H, SEG_W), device=device, dtype=hn_in.dtype)
        seg_canvas[:, :, :PROC_H, :PROC_W] = hn_in
        seg_in = seg_canvas
        det_in = small_bgr.div_(255.0).to(device)
    else:
        small_bgr = [cv2.resize(f, (PROC_W, PROC_H)) for f in frames]
        small_rgb = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in small_bgr]
        seg_list = []
        for rgb_img in small_rgb:
            canvas = np.zeros((SEG_H, SEG_W, 3), np.uint8)
            canvas[:PROC_H, :PROC_W] = rgb_img
            seg_list.append(hn_pre_cpu(canvas))
        seg_in = torch.stack(seg_list)
        det_in = torch.stack([
            torch.from_numpy(b.transpose(2, 0, 1)).float().div_(255.0)
            for b in small_bgr
        ]).to(device)

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
        try:
            dets = yolop_nms(det_flat, conf_thres=0.7, iou_thres=0.4)
        except NotImplementedError:
            det_flat_cpu = det_flat.cpu()
            dets = yolop_nms(det_flat_cpu, conf_thres=0.7, iou_thres=0.4)

    masks_small = seg.argmax(1).cpu().numpy()[:, :PROC_H, :PROC_W]
    sx, sy = orig_W / PROC_W, orig_H / PROC_H
    scale = orig_H / 720.0
    fnt_scale = 0.7 * scale
    t_bg, t_fg = max(2, int(3*scale)), max(1, int(1.2*scale))
    scale_tensor = torch.tensor([sx, sy, sx, sy], device='cpu')

    matched_tracks = set()

    for idx, det in enumerate(dets):
        frame = frames[idx]
        mk = masks_small[idx]
        mask = cv2.resize(mk.astype(np.uint8), (orig_W, orig_H), cv2.INTER_NEAREST)
        overlay = frame.copy()
        overlay[mask == 2] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        depth_map = depth_maps[idx]

        if det is not None and len(det):
            if det.device.type != 'cpu':
                det = det.cpu()
            det = det.clone()
            det[:, :4] *= scale_tensor
            for x1, y1, x2, y2, conf, cls in det.numpy():
                if conf < 0.45:
                    continue
                xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))

                # Sample median depth over small patch
                x_center = (xi1 + xi2) / 2.0
                y_bottom = yi2
                x_proc = int(x_center / sx)
                y_proc = int(y_bottom / sy)
                x0 = max(0, x_proc - 2)
                x1p = min(PROC_W - 1, x_proc + 2)
                y0 = max(0, y_proc - 2)
                y1p = min(PROC_H - 1, y_proc + 2)
                patch = depth_map[y0:y1p+1, x0:x1p+1]
                if patch.size > 0:
                    depth_val = float(np.median(patch))
                else:
                    depth_val = float(depth_map[y_proc, x_proc])

                # Adaptive smoothing: match track
                matched = None
                for ti, tr in enumerate(depth_tracks):
                    if tr['class'] == int(cls) and abs(tr['x'] - x_center) < TRACK_X_THRESH:
                        matched = (ti, tr)
                        break
                if matched is not None:
                    ti, tr = matched
                    old_depth = tr['depth']
                    alpha_old = ALPHA_OLD_APPROACH if depth_val < old_depth else ALPHA_OLD_AWAY
                    sm = alpha_old * old_depth + (1 - alpha_old) * depth_val
                    tr['depth'] = sm
                    tr['x'] = x_center
                    tr['miss'] = 0
                    depth_disp = sm
                    matched_tracks.add(ti)
                else:
                    depth_disp = depth_val
                    depth_tracks.append({'class': int(cls), 'x': x_center, 'depth': depth_val, 'miss': 0})
                    matched_tracks.add(len(depth_tracks)-1)

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
                # confidence (bottom line)
                for col in [(0,0,0), (255,0,255)]:
                    cv2.putText(vis, conf_text, (x_text, y_line1),
                                cv2.FONT_HERSHEY_SIMPLEX, ts, col, t_bg if col==(0,0,0) else t_fg)
                # depth (above)
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
