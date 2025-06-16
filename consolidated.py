"""
YOLOP demo – object detection + smoothed drivable-area overlay
"""

import time, queue, threading, logging, cv2, numpy as np, torch, sys
from pathlib import Path

# ── config ──
VIDEO_IN   = r"C:\Users\jiami\Videos\2025-06-15_17-01-53-front.mp4"
VIDEO_OUT  = "yolop_drive_out.avi"
IMG_SIZE   = 640
BATCH      = 8
CONF_THRES = 0.4
IOU_THRES  = 0.45
TARGET_FPS = 30
MASK_THR   = 0.55                # ← tighter threshold
# ───────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
logging.getLogger("torch").setLevel(logging.ERROR)

# patch torchvision NMS → CPU
import torchvision.ops as _ops
_orig_nms = _ops.nms
_ops.nms  = lambda b,s,iou: _orig_nms(b.cpu(), s.cpu(), iou)

# load YOLOP
yolop = torch.hub.load("hustvl/yolop", "yolop",
                       pretrained=True).to(device).eval().half()

_repo = Path(torch.hub.get_dir()) / "hustvl_yolop_main" / "lib"
sys.path.append(str(_repo))
from core.general import non_max_suppression

# video I/O --------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_IN);  assert cap.isOpened()
fps_src = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
ret, first = cap.read();           assert ret
H0, W0 = first.shape[:2]

writer = cv2.VideoWriter(
    VIDEO_OUT, cv2.VideoWriter_fourcc(*"XVID"), fps_src, (W0, H0)
)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# threaded queues --------------------------------------------------------
f_q, r_q = queue.Queue(BATCH*2), queue.Queue(BATCH*2)
_SENTINEL = object()

def reader():
    while True:
        ok, frm = cap.read()
        if not ok:
            f_q.put(_SENTINEL);  return
        f_q.put(frm)

@torch.inference_mode()
def process_batch(frames):
    # prepare batch
    imgs = [
        torch.from_numpy(
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                       (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)
        ).permute(2,0,1).float()/255.0
        for f in frames
    ]
    batch = torch.stack(imgs).to(device).half()

    det_raw, da_seg_out, _ = yolop(batch)
    det_list = non_max_suppression(det_raw[0], CONF_THRES, IOU_THRES)

    prob_maps = torch.softmax(da_seg_out.float(), 1)[:,1].cpu().numpy()

    outs=[]
    for frm, det, prob in zip(frames, det_list, prob_maps):
        h, w = frm.shape[:2]

        mk = cv2.resize(prob.astype(np.float32), (w, h), cv2.INTER_LINEAR)
        mk = (mk > MASK_THR).astype(np.uint8)*255
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))  # ← smaller
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))  # ← smaller

        overlay = np.zeros_like(frm);  overlay[mk>128] = (0,255,0)
        vis = cv2.addWeighted(frm, 0.8, overlay, 0.2, 0)   # a bit less tint

        if det is not None and len(det):
            det = det.cpu()
            sx, sy = w/IMG_SIZE, h/IMG_SIZE
            det[:,[0,2]] *= sx; det[:,[1,3]] *= sy
            for *xyxy, conf, cls in det.numpy():
                x1,y1,x2,y2 = map(int, xyxy)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(255,0,255),2)
                cv2.putText(vis,f"{int(cls):d}:{conf:.2f}",
                            (x1,max(y1-5,0)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,
                            (255,255,255),2,cv2.LINE_AA)
        outs.append(vis)
    return outs

def worker():
    batch=[]
    while True:
        frm = f_q.get()
        if frm is _SENTINEL:
            if batch:
                for v in process_batch(batch): r_q.put(v)
            r_q.put(_SENTINEL);  return
        batch.append(frm)
        if len(batch) == BATCH:
            for v in process_batch(batch): r_q.put(v)
            batch=[]

threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

# paced display ----------------------------------------------------------
frame_dt = 1.0/TARGET_FPS;  last_t = time.time()
print("running – press q to quit")

while True:
    frm = r_q.get()
    if frm is _SENTINEL: break
    delay = frame_dt - (time.time() - last_t)
    if delay > 0: time.sleep(delay)
    last_t = time.time()

    writer.write(frm)
    cv2.imshow("Output", frm)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); writer.release(); cv2.destroyAllWindows()
