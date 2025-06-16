"""
YOLOP  •  object-detection-only  •  larger plain-white labels
"""

import cv2, time, queue, threading, logging, numpy as np, torch, sys
from pathlib import Path

# ─── config ──────────────────────────────────────────────────
VIDEO_IN   = r"C:\Users\jiami\Videos\2025-06-15_17-01-53-front.mp4"
VIDEO_OUT  = "yolop_objects_only.avi"
IMG_SIZE   = 640
BATCH      = 8
CONF_THRES = 0.6
IOU_THRES  = 0.50
TARGET_FPS = 30
# ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
logging.getLogger("torch").setLevel(logging.ERROR)

# torchvision NMS → CPU
import torchvision.ops as _ops
_orig_nms = _ops.nms
_ops.nms  = lambda b,s,iou: _orig_nms(b.cpu(), s.cpu(), iou)

# load YOLOP
yolop = torch.hub.load("hustvl/yolop", "yolop",
                       pretrained=True).to(device).eval().half()

# repo helper for NMS
_repo = Path(torch.hub.get_dir()) / "hustvl_yolop_main" / "lib"
sys.path.append(str(_repo))
from core.general import non_max_suppression

CLS_NAME = ["car","bus","truck","person","bicycle",
            "motorbike","trafficLight","trafficSign","train","rider"]

# ─── video I/O ───────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_IN);  assert cap.isOpened()
fps_src = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
ret, first = cap.read();           assert ret
H0, W0 = first.shape[:2]

writer = cv2.VideoWriter(
    VIDEO_OUT, cv2.VideoWriter_fourcc(*"XVID"), fps_src, (W0, H0)
)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# ─── queues & threads ───────────────────────────────────────
f_q, r_q = queue.Queue(BATCH*2), queue.Queue(BATCH*2)
_SENTINEL = object()
def reader():
    while True:
        ok, frm = cap.read()
        if not ok:
            f_q.put(_SENTINEL); return
        f_q.put(frm)
threading.Thread(target=reader, daemon=True).start()

@torch.inference_mode()
def process(frames):
    imgs = [torch.from_numpy(
                cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                           (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)
            ).permute(2,0,1).float()/255.0 for f in frames]
    batch = torch.stack(imgs).to(device).half()

    det_raw, *_ = yolop(batch)
    dets = non_max_suppression(det_raw[0], CONF_THRES, IOU_THRES)

    outs=[]
    for f_orig, det in zip(frames, dets):
        h,w = f_orig.shape[:2]
        if det is not None and len(det):
            det = det.cpu()
            sx, sy = w/IMG_SIZE, h/IMG_SIZE
            det[:,[0,2]] *= sx;  det[:,[1,3]] *= sy
            for *xyxy, conf, cls in det.numpy():
                x1,y1,x2,y2 = map(int,xyxy)
                cv2.rectangle(f_orig,(x1,y1),(x2,y2),(255,0,255),2)
                label = f"{CLS_NAME[int(cls)]}:{conf:.2f}"
                cv2.putText(f_orig, label,
                            (x1, max(y1-7, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,255,255), 3, cv2.LINE_AA)  # larger font
        outs.append(f_orig)
    return outs

def worker():
    batch=[]
    while True:
        frm = f_q.get()
        if frm is _SENTINEL:
            if batch:
                for v in process(batch): r_q.put(v)
            r_q.put(_SENTINEL); return
        batch.append(frm)
        if len(batch) == BATCH:
            for v in process(batch): r_q.put(v)
            batch.clear()
threading.Thread(target=worker, daemon=True).start()

# ─── display loop ────────────────────────────────────────────
frame_dt = 1.0 / TARGET_FPS
last = time.time()
print("running – press q to quit")

while True:
    vis = r_q.get()
    if vis is _SENTINEL: break
    delay = frame_dt - (time.time() - last)
    if delay > 0: time.sleep(delay)
    last = time.time()

    writer.write(vis);  cv2.imshow("Output", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); writer.release(); cv2.destroyAllWindows()
