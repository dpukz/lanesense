"""
YOLOP single-network demo: object detection + drivable-area overlay
– patched so torchvision.ops.nms always runs on CPU
"""
import time, queue, threading, logging, cv2, numpy as np, torch, sys
from pathlib import Path

# ────── basic settings ──────
VIDEO_IN   = r"C:\Users\jiami\Videos\2025-06-15_17-01-53-front.mp4"
VIDEO_OUT  = "yolop_drive_out.avi"
IMG_SIZE   = 640          # YOLOP native square
BATCH      = 8
CONF_THRES = 0.25
IOU_THRES  = 0.45
TARGET_FPS = 30
# ────────────────────────────

# silence hub / torch chatter
logging.getLogger("torch").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ────── PATCH torchvision NMS → CPU only ──────
import torchvision.ops as _ops
_orig_nms = _ops.nms
def _cpu_nms(boxes, scores, iou_th):
    return _orig_nms(boxes.cpu(), scores.cpu(), iou_th)
_ops.nms = _cpu_nms
# ──────────────────────────────────────────────

# load YOLOP
yolop = torch.hub.load("hustvl/yolop", "yolop",
                       pretrained=True).to(device).eval().half()

# repo utils (scale_coords / NMS wrapper)
_repo = Path(torch.hub.get_dir()) / "hustvl_yolop_main" / "lib"
sys.path.append(str(_repo))
from core.general import non_max_suppression, scale_coords

# video I/O
cap = cv2.VideoCapture(VIDEO_IN);  assert cap.isOpened()
fps_src = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
ret, first = cap.read();           assert ret
H0, W0 = first.shape[:2]

writer = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps_src, (W0, H0)
)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# queues
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
    imgs = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        lb  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(lb).permute(2,0,1).float() / 255.0
        imgs.append(img)
    batch = torch.stack(imgs).to(device).half()

    det_raw, da_seg, _ = yolop(batch)
    det_list = non_max_suppression(det_raw[0], CONF_THRES, IOU_THRES)
    da_masks = da_seg.argmax(1).byte().cpu().numpy()

    outs = []
    for frm, det, mask_small in zip(frames, det_list, da_masks):
        h, w = frm.shape[:2]
        mask = cv2.resize(mask_small, (w, h), cv2.INTER_NEAREST)
        overlay = np.zeros_like(frm); overlay[mask==1] = (0,255,0)
        vis = cv2.addWeighted(frm, 0.7, overlay, 0.3, 0)

        if det is not None and len(det):
            det[:,:4] = scale_coords(batch.shape[2:], det[:,:4], frm.shape).round()
            for *xyxy, conf, cls in det.cpu().numpy():
                x1,y1,x2,y2 = map(int, xyxy)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,255), 2)
                cv2.putText(vis, f"{int(cls):d}:{conf:.2f}",
                            (x1, max(y1-5,0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255,255,255), 2, cv2.LINE_AA)
        outs.append(vis)
    return outs

def worker():
    batch=[]
    while True:
        frm = f_q.get()
        if frm is _SENTINEL:
            if batch:
                for v in process_batch(batch): r_q.put(v)
            r_q.put(_SENTINEL)
            return
        batch.append(frm)
        if len(batch) == BATCH:
            for v in process_batch(batch): r_q.put(v)
            batch = []

threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

# paced display
frame_dt = 1.0 / TARGET_FPS
last_t   = time.time()
print("running – press q to quit")

while True:
    frm = r_q.get()
    if frm is _SENTINEL:
        break
    delay = frame_dt - (time.time() - last_t)
    if delay > 0:
        time.sleep(delay)
    last_t = time.time()

    writer.write(frm)
    cv2.imshow("Output", frm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); writer.release(); cv2.destroyAllWindows()
