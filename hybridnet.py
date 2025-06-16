import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import cv2, sys, time, threading, queue
import torch, torch.nn.functional as F, numpy as np, torchvision
from ultralytics import YOLO

# ------------------ patch torchvision NMS to CPU-only ------------------
import torchvision.ops as _ops
_orig_nms = _ops.nms
def _cpu_nms(boxes, scores, iou):
    return _orig_nms(boxes.cpu(), scores.cpu(), iou)
_ops.nms = _cpu_nms
# -----------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --- HybridNets for lane segmentation (GPU) ---
hn = torch.hub.load(
    "datvuthanh/hybridnets", "hybridnets",
    pretrained=True, trust_repo=True
).to(device).eval().half()

# --- YOLOv5n for object detection (GPU inference) ---
yolo = YOLO("yolov5n.pt")      # auto-downloads
yolo.model.to(device).eval()

# -----------------------------------------------------------------------
VIDEO_IN  = r"C:\Users\jiami\Downloads\yt1z.net - 3-Hour Drive on California Freeway Dash Cam Tours (360p).mp4"
VIDEO_OUT = "output_hybridnets_yolo.avi"
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open video"
ok, f0 = cap.read(); assert ok, "Cannot read first frame"
orig_H, orig_W = f0.shape[:2]
CROP_Y = orig_H // 2                     # only detect objects in bottom half
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
out = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (orig_W, orig_H)
)

# -----------------------------------------------------------------------
# HybridNets preprocessing
IN_W, IN_H = 640, 384
_hn_mean = np.array([.485,.456,.406],np.float32)
_hn_std  = np.array([.229,.224,.225],np.float32)
def hn_preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sml = cv2.resize(rgb, (IN_W, IN_H))
    x   = (sml.astype(np.float32)/255. - _hn_mean)/_hn_std
    return torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)

# ↑↑ ROI-crop optimization here ↑↑
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
    fps_inst = 1.0/((now-prev_t)/len(frames))
    prev_t   = now

    # --- HybridNets lane mask (full frame) ---
    x = torch.cat([hn_preprocess(f) for f in frames]).half()
    with torch.inference_mode():
        _,_,_,_,seg = hn(x)
    mask = F.interpolate(
        seg.argmax(1,keepdim=True).float(),
        size=(orig_H,orig_W),
        mode="nearest"
    ).long()[:,0]  # (B,H,W)

    # --- prepare cropped frames for YOLO ---
    cropped = [f[CROP_Y:orig_H, :] for f in frames]

    # --- Batched YOLOv5n on cropped regions (GPU) ---
    with torch.inference_mode():
        detections = yolo(
            cropped,
            device=device,
            show=False,
            verbose=False,
            conf=0.2,
            iou=0.15,
            imgsz=640
        )

    # --- render each frame ---
    for frame, mk, det in zip(frames, mask, detections):
        # lane overlay
        mk_np = mk.cpu().numpy().astype(np.uint8)
        overlay = frame.copy()
        overlay[mk_np==2] = (0,255,0)
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # draw YOLO boxes (shift Y by CROP_Y)
        xyxy   = det.boxes.xyxy.cpu().numpy()
        confs  = det.boxes.conf.cpu().numpy()
        cls_id = det.boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2), c, cl in zip(xyxy, confs, cls_id):
            if c < 0.5: continue
            y1, y2 = int(y1 + CROP_Y), int(y2 + CROP_Y)
            x1, x2 = int(x1), int(x2)
            lbl = yolo.names[cl]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.putText(vis, f"{lbl}:{c:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)

        # FPS overlay
        cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
        cv2.putText(vis, f"{fps_inst:4.1f} FPS", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)

        result_q.put(vis)

    return prev_t

def worker():
    prev, batch = time.time(), []
    while True:
        frm = frame_q.get()
        if frm is _SENTINEL:
            if batch: prev = process_batch(batch, prev)
            result_q.put(_SENTINEL)
            return
        batch.append(frm)
        if len(batch) < BATCH: continue
        prev = process_batch(batch, prev)
        batch = []

# start threads
threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

# display & write
cv2.namedWindow("HybridNets + YOLOv5n")
slider_updating = False
def on_trackbar(pos):
    if slider_updating: return
    with cap_lock:
        cap.set(cv2.CAP_PROP_POS_MSEC, pos*100)

total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps*10)
cv2.createTrackbar("Time [...s]", "HybridNets + YOLOv5n", 0, total_sec, on_trackbar)

TARGET_FPS, FRAME_PERIOD = 30.0, 1.0/30.0
last = time.time()

print("running – press q to quit")
while True:
    vis = result_q.get()
    if vis is _SENTINEL: break

    now = time.time()
    if now - last < FRAME_PERIOD:
        time.sleep(FRAME_PERIOD - (now - last))
    last = time.time()

    out.write(vis)
    cv2.imshow("HybridNets + YOLOv5n", vis)

    with cap_lock:
        pos = int(cap.get(cv2.CAP_PROP_POS_MSEC)/100)
    slider_updating = True
    cv2.setTrackbarPos("Time [...s]", "HybridNets + YOLOv5n", pos)
    slider_updating = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
