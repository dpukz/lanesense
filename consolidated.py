"""
YOLOP – ego-lane highlighting + object boxes
"""

import time, queue, threading, logging, cv2, numpy as np, torch, sys
from pathlib import Path

# ─── config ────────────────────────────────────────────────────────────
VIDEO_IN   = r"C:\Users\jiami\Videos\2025-06-15_17-01-53-front.mp4"
VIDEO_OUT  = "yolop_ego_lane.avi"
IMG_SIZE   = 640
BATCH      = 8
CONF_THRES = 0.25
IOU_THRES  = 0.45
TARGET_FPS = 30
ROAD_THR   = 0.50          # drivable-prob threshold
LANE_THR   = 0.40          # lane-prob threshold
# ───────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
logging.getLogger("torch").setLevel(logging.ERROR)

# ─── torchvision NMS → CPU (compile-agnostic) ──────────────────────────
import torchvision.ops as _ops
_orig_nms = _ops.nms
_ops.nms  = lambda b,s,iou: _orig_nms(b.cpu(), s.cpu(), iou)
# ───────────────────────────────────────────────────────────────────────

# load YOLOP
yolop = torch.hub.load("hustvl/yolop", "yolop",
                       pretrained=True).to(device).eval().half()

# repo util for NMS wrapper
_repo = Path(torch.hub.get_dir()) / "hustvl_yolop_main" / "lib"
sys.path.append(str(_repo))
from core.general import non_max_suppression

# ─── video I/O ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_IN);  assert cap.isOpened()
fps_src = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
ret, first = cap.read();           assert ret
H0, W0 = first.shape[:2]

writer = cv2.VideoWriter(
    VIDEO_OUT, cv2.VideoWriter_fourcc(*"XVID"), fps_src, (W0, H0)
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

# helper: build binary ego-lane mask between centre-straddling lane lines
def ego_lane_from_lines(lane_mask, w):
    h = lane_mask.shape[0]
    rows = np.linspace(h-1, int(h*0.5), 30, dtype=int)    # bottom-half samples
    centre = w // 2
    left_pts, right_pts = [], []

    for y in rows:
        xs = np.where(lane_mask[y] > 0)[0]
        if xs.size < 2: continue
        left_side  = xs[xs < centre]
        right_side = xs[xs > centre]
        if left_side.size and right_side.size:
            left_pts.append((left_side.max(),  y))
            right_pts.append((right_side.min(), y))

    if len(left_pts) < 4 or len(right_pts) < 4:
        return None                       # not confident this frame

    poly = np.array(left_pts + right_pts[::-1], np.int32)
    mask = np.zeros_like(lane_mask, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

@torch.inference_mode()
def process_batch(frames):
    # prepare square RGB batch
    imgs = [
        torch.from_numpy(
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                       (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)
        ).permute(2,0,1).float() / 255.0
        for f in frames
    ]
    batch = torch.stack(imgs).to(device).half()

    det_raw, da_seg, ll_seg = yolop(batch)
    det_list = non_max_suppression(det_raw[0], CONF_THRES, IOU_THRES)

    road_prob = torch.softmax(da_seg.float(), 1)[:, 1].cpu().numpy()   # (B,h,w)
    lane_prob = torch.softmax(ll_seg.float(), 1)[:, 1:].sum(1).cpu().numpy()  # any lane

    outs = []
    for frm, det, rd_p, ln_p in zip(frames, det_list, road_prob, lane_prob):
        h, w = frm.shape[:2]

        # --- masks at half-res for speed --------------------------------
        h2, w2 = h//2, w//2
        rd = cv2.resize(rd_p.astype(np.float32), (w2, h2), cv2.INTER_LINEAR)
        ln = cv2.resize(ln_p.astype(np.float32), (w2, h2), cv2.INTER_LINEAR)

        rd_mask = (rd > ROAD_THR).astype(np.uint8)*255
        ln_mask = (ln > LANE_THR).astype(np.uint8)

        ego_half = ego_lane_from_lines(ln_mask, w2)
        if ego_half is None:
            ego_half = rd_mask                    # fallback: whole road
        else:
            ego_half = cv2.bitwise_and(rd_mask, ego_half)

        # small morphology then upscale
        ego_half = cv2.morphologyEx(ego_half, cv2.MORPH_CLOSE,
                                    np.ones((5,5), np.uint8))
        ego = cv2.resize(ego_half, (w, h), cv2.INTER_LINEAR)

        overlay = np.zeros_like(frm)
        overlay[ego > 128] = (0, 255, 0)
        vis = cv2.addWeighted(frm, 0.8, overlay, 0.2, 0)

        # boxes
        if det is not None and len(det):
            det = det.cpu()
            sx, sy = w / IMG_SIZE, h / IMG_SIZE
            det[:, [0, 2]] *= sx
            det[:, [1, 3]] *= sy
            for *xyxy, conf, cls in det.numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(vis, f"{int(cls):d}:{conf:.2f}",
                            (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

        outs.append(vis)
    return outs

def worker():
    batch = []
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

# paced display loop
frame_dt = 1.0 / TARGET_FPS
last_t = time.time()
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
