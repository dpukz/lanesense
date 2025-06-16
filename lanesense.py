import math
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import cv2, time, threading, queue
import torch, torch.nn.functional as F
import numpy as np
from ultralytics import YOLO

# ------------------ patch torchvision NMS to CPU-only ------------------
import torchvision.ops as _ops
_orig_nms = _ops.nms
def _cpu_nms(boxes, scores, iou):
    return _orig_nms(boxes.cpu(), scores.cpu(), iou)
_ops.nms = _cpu_nms
# -----------------------------------------------------------------------

PROC_W, PROC_H = 640, 360
SEG_W = math.ceil(PROC_W/32)*32
SEG_H = math.ceil(PROC_H/32)*32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

hn = torch.hub.load(
    "datvuthanh/hybridnets", "hybridnets",
    pretrained=True, trust_repo=True
).to(device).eval().half()
yolo = YOLO("yolov5n.pt")
yolo.model.to(device).eval()

VIDEO_IN  = r"C:\Users\jiami\Videos\2025-06-15_17-01-53-front.mp4"
VIDEO_OUT = "output_downscaled_fix.avi"
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open video"

ret, first = cap.read()
assert ret, "Cannot read first frame"
orig_H, orig_W = first.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

out = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (orig_W, orig_H)
)

cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

_hn_mean = np.array([.485, .456, .406], np.float32)
_hn_std  = np.array([.229, .224, .225], np.float32)
def seg_preprocess(rgb_pad):
    x = (rgb_pad.astype(np.float32)/255.0 - _hn_mean) / _hn_std
    x = x.transpose(2,0,1)
    return torch.from_numpy(x).unsqueeze(0).to(device).half()

BATCH = 8
frame_q, result_q = queue.Queue(BATCH*2), queue.Queue(BATCH*2)
_SENTINEL = object()
cap_lock = threading.Lock()

def reader():
    while True:
        with cap_lock:
            ret, frame = cap.read()
        if not ret:
            frame_q.put(_SENTINEL)
            return
        frame_q.put(frame)

def process_batch(frames, prev_t):
    now      = time.time()
    fps_inst = 1.0/((now - prev_t)/len(frames))
    prev_t   = now

    smalls_bgr = [cv2.resize(f, (PROC_W, PROC_H)) for f in frames]
    smalls_rgb = [cv2.cvtColor(s, cv2.COLOR_BGR2RGB) for s in smalls_bgr]
    padded = []
    for rgb in smalls_rgb:
        pad = np.zeros((SEG_H, SEG_W, 3), dtype=np.uint8)
        pad[:PROC_H, :PROC_W] = rgb
        padded.append(pad)

    x = torch.cat([seg_preprocess(p) for p in padded], dim=0)
    with torch.inference_mode():
        _,_,_,_,seg = hn(x)
    mask_padded = seg.argmax(1).cpu().numpy().astype(np.uint8)
    masks_small = [mp[:PROC_H, :PROC_W] for mp in mask_padded]

    with torch.inference_mode():
        dets = yolo(
            smalls_bgr,
            device=device,
            show=False,
            verbose=False,
            conf=0.18,
            iou=0.15,
            imgsz=PROC_W
        )

    sx, sy = orig_W/PROC_W, orig_H/PROC_H
    scale = orig_H / 720
    fnt_scale = 0.7 * scale
    thickness_bg = max(2, int(3 * scale))
    thickness_fg = max(1, int(1.2 * scale))

    for orig, mk_s, det in zip(frames, masks_small, dets):
        mk = cv2.resize(mk_s, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
        overlay = orig.copy()
        overlay[mk==2] = (0,255,0)
        vis = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)

        for (x1,y1,x2,y2), c, cl in zip(
                det.boxes.xyxy.cpu().numpy(),
                det.boxes.conf.cpu().numpy(),
                det.boxes.cls.cpu().numpy().astype(int)
            ):
            if c < 0.5: continue
            xi1, yi1 = int(x1*sx), int(y1*sy)
            xi2, yi2 = int(x2*sx), int(y2*sy)
            lbl = yolo.names[cl]
            label_text = f"{lbl}:{c:.2f}"
            cv2.rectangle(vis, (xi1,yi1), (xi2,yi2), (255,0,255), max(1, int(2 * scale)))
            cv2.putText(vis, label_text, (xi1, max(yi1-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, (0, 0, 0), thickness_bg)
            cv2.putText(vis, label_text, (xi1, max(yi1-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, (255, 0, 255), thickness_fg)

        cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10, int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, (0, 0, 0), thickness_bg)
        cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10, int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fnt_scale, (255, 255, 255), thickness_fg)

        result_q.put(vis)

    return prev_t

def worker():
    prev = time.time()
    batch = []
    while True:
        frm = frame_q.get()
        if frm is _SENTINEL:
            if batch:
                prev = process_batch(batch, prev)
            result_q.put(_SENTINEL)
            return
        batch.append(frm)
        if len(batch) >= BATCH:
            prev = process_batch(batch, prev)
            batch = []

threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

print("running – press q to quit")
while True:
    vis = result_q.get()
    if vis is _SENTINEL:
        break
    out.write(vis)
    cv2.imshow("Output", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()