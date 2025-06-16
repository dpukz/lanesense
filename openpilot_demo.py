#!/usr/bin/env python3
"""
lane_detect_yolop.py — Real-time Ego‐Lane Path Detection (GPU forced)
────────────────────────────────────────────────────
• python lane_detect_yolop.py                 ← webcam 0
• python lane_detect_yolop.py "clip.mp4"      ← video file
"""
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F

# ─── FORCE GPU ─────────────────────────────────────────────────────────
device = 'cuda:0'
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available — install CUDA‐enabled PyTorch & NVIDIA drivers.")
print(f"[INFO] Using device: {device.upper()}")

# ─── YOLOP MODEL SETUP ──────────────────────────────────────────────────
MODEL_REPO = 'hustvl/yolop'
MODEL_NAME = 'yolop'

def load_model(device, half=True):
    print(f"[INFO] Loading YOLOP model on {device}...")
    model = torch.hub.load(MODEL_REPO, MODEL_NAME, pretrained=True, trust_repo=True)
    model.to(device).eval()
    if half:
        model.half()
        print("[INFO] Model in FP16 (half‐precision) mode.")
    return model

# ─── PREPARE FRAME (LETTERBOX) ─────────────────────────────────────────
def prepare_frame(img, size=640):
    """Resize & pad to square for YOLOP input, preserving aspect ratio."""
    h,w = img.shape[:2]
    scale = size/float(max(h,w))
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - nw, size - nh
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                borderType=cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, scale, pad_w, pad_h

# ─── VIDEO PROCESSING LOOP ──────────────────────────────────────────────
def process_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {source}"); return

    model = load_model(device=device, half=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_h, orig_w = frame.shape[:2]

        # 1) Letterbox → 640×640
        img_pad, _, pad_w, pad_h = prepare_frame(frame, size=640)
        rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2,0,1).float().div(255.0).unsqueeze(0)
        tensor = tensor.to(device).half()  # FP16 on GPU

        # 2) Inference → detection, drivable_area, lane_line
        with torch.no_grad():
            _, da_out, ll_out = model(tensor)

        # 3) Remove padding from each seg mask
        new_h, new_w = img_pad.shape[0]-pad_h, img_pad.shape[1]-pad_w
        da_crop = da_out[:, :, :new_h, :new_w]
        ll_crop = ll_out[:, :, :new_h, :new_w]

        # 4) Upsample back to original resolution
        da_up = F.interpolate(da_crop, size=(orig_h,orig_w), mode='bilinear', align_corners=False)
        ll_up = F.interpolate(ll_crop, size=(orig_h,orig_w), mode='bilinear', align_corners=False)

        # 5) Argmax → binary masks
        da_mask = da_up.argmax(dim=1).squeeze().cpu().byte().numpy()
        ll_mask = ll_up.argmax(dim=1).squeeze().cpu().byte().numpy()

        # 6) Morphological closing to fill small holes (especially at bottom)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        da_mask = cv2.morphologyEx(da_mask, cv2.MORPH_CLOSE, kernel)

        # 7) Find left/right lane boundaries at bottom
        y0 = orig_h - 5
        xs = np.where(ll_mask[y0] == 1)[0]
        if xs.size >= 2:
            x_min, x_max = xs.min(), xs.max()
        else:
            x_min, x_max = 0, orig_w  # fallback = entire width

        # 8) Restrict drivable area to between the two lane lines
        ego_lane = np.zeros_like(da_mask)
        ego_lane[:, x_min:x_max+1] = da_mask[:, x_min:x_max+1]

        # 9) Overlay coloring
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[ego_lane == 1] = (0,255,0)
        output = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

        # 10) Display
        cv2.imshow("Ego Lane Path (GPU)", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── MAIN ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv)>1 else 0
    process_video(source=src)
