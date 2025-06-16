import cv2
import torch
import numpy as np
from collections import deque
import argparse

# Parse command-line arguments for input video
parser = argparse.ArgumentParser(description="Ego-Lane Detection and Path Shading using YOLOP and MiDaS")
parser.add_argument("--input", "-i", type=str, default=None,
                    help="Path to input video file. If not provided, webcam (device 0) will be used.")
args = parser.parse_args()

# Select device: force GPU if available, otherwise use CPU
assert torch.cuda.is_available(), "CUDA GPU not available. Please check your PyTorch and GPU setup."
device = torch.device("cuda")

print(f"[INFO] Using device: {device}")

# Load YOLOP model from PyTorch Hub (pretrained on BDD100K)
print("[INFO] Loading YOLOP model from PyTorch Hub...")
yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
yolop_model.to(device)
yolop_model.eval()

# Load MiDaS (DPT-Large) model from PyTorch Hub:contentReference[oaicite:9]{index=9}
print("[INFO] Loading MiDaS (DPT-Large) model from PyTorch Hub...")
midas_model_type = "DPT_Large"  # highest accuracy model
midas_model = torch.hub.load("intel-isl/MiDaS", midas_model_type, pretrained=True)
midas_model.to(device)
midas_model.eval()

# Load MiDaS transforms for input preprocessing:contentReference[oaicite:10]{index=10}
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if midas_model_type in ["DPT_Large", "DPT_Hybrid"]:
    midas_transform = midas_transforms.dpt_transform
else:
    midas_transform = midas_transforms.small_transform

# Function to letterbox (pad) an image to the target size while keeping aspect ratio
def letterbox(img, size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    target_h, target_w = size
    # Calculate scale and padding
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Resize image with aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create new image of desired size and fill with padding color
    result = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    # Compute top-left corner for centering
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return result, scale, pad_x, pad_y

# Initialize video source (webcam or file)
if args.input is None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. If you intended to use a video file, please specify --input.")
    print("[INFO] Running on webcam. Press 'q' to quit.")
else:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {args.input}")
    print(f"[INFO] Processing video file {args.input}")

# Prepare smoothing buffers for left and right lane lines (store polynomial coefficients or points)
history_length = 5  # number of frames to average over for smoothing
left_history = deque(maxlen=history_length)
right_history = deque(maxlen=history_length)

# Choose a color and transparency for lane shading
overlay_color = (0, 255, 0)     # green color for lane overlay (BGR format for OpenCV)
overlay_alpha = 0.4             # transparency factor (0 = fully transparent, 1 = opaque)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video or cannot fetch frame
    orig_frame = frame.copy()
    h, w = frame.shape[:2]

    # 1. Preprocess frame for YOLOP (letterbox to 640x640 and convert to tensor)
    img_640, scale, pad_x, pad_y = letterbox(frame, size=(640, 640))
    # Convert BGR (OpenCV) to RGB for model, then to tensor
    img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0  # normalize 0-255 to 0-1
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)    # shape (1,3,640,640)
    img_tensor = img_tensor.to(device)

    # 2. Run YOLOP model to get lane segmentation
    with torch.no_grad():
        # YOLOP outputs: det_out (detections), da_seg_out (drivable area mask), ll_seg_out (lane line mask)
        det_out, da_seg_out, ll_seg_out = yolop_model(img_tensor)
    # Get lane line mask (assuming output is (1,2,H,W) with [background, lane] or vice versa)
    # We use argmax to get binary mask of lane vs background
    ll_seg = ll_seg_out[0]  # shape (2, H, W)
    # If YOLOP output channels are [background, lane], argmax gives lane=1 where lane probability > background
    lane_mask_640 = (ll_seg.argmax(dim=0).cpu().numpy().astype(np.uint8))  # shape (H, W), values 0 or 1
    # Crop out padding to original image size
    # Calculate dimensions of unpadded region in the 640x640 image
    new_h = int(h * scale)
    new_w = int(w * scale)
    # Region in padded image corresponding to original image
    x1, x2 = pad_x, pad_x + new_w
    y1, y2 = pad_y, pad_y + new_h
    # Resize mask back to original frame resolution
    if new_h > 0 and new_w > 0:
        lane_mask = lane_mask_640[y1:y2, x1:x2]  # unpad
        lane_mask = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # In case scaling went wrong (should not usually happen), fallback to direct resize
        lane_mask = cv2.resize(lane_mask_640, (w, h), interpolation=cv2.INTER_NEAREST)
    # lane_mask is now a binary (0/1) mask at original frame size

    # 3. Run MiDaS model to get depth (inverse depth map)
    # Apply MiDaS input transform (which resizes and normalizes for the model)
    input_batch = midas_transform(orig_frame).to(device)
    with torch.no_grad():
        depth_pred = midas_model(input_batch)
        # Resize depth to original frame size
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=(h, w),  # (height, width)
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = depth_pred.cpu().numpy()
    # Note: depth_map is an inverse depth (higher value = closer camera, lower = farther).

    # 4. Depth-aware filtering of lane mask.
    # Remove lane mask pixels that are very far or not on ground by thresholding depth.
    # We determine a depth cutoff based on the median depth of the lower half of the image (road region).
    lower_half = depth_map[h//2:, :]  # bottom half of frame (likely road region in front of car)
    if lower_half.size != 0:
        # Use median + a factor as threshold for far distance
        depth_cutoff = np.median(lower_half) * 0.4  # adjust 0.4 factor if needed to be stricter or looser
    else:
        depth_cutoff = np.median(depth_map) * 0.4
    # Filter: keep lane mask where depth_map is above cutoff (meaning closer than a certain distance)
    # Because MiDaS outputs higher values for closer objects (inverse depth),
    # a small cutoff retains only very close areas; we want to remove extremely far or inconsistent points.
    depth_filter = (depth_map > depth_cutoff).astype(np.uint8)
    lane_mask = cv2.bitwise_and(lane_mask, lane_mask, mask=depth_filter)

    # 5. Identify lane boundary lines using connected components on the filtered mask
    left_line_poly = None
    right_line_poly = None
    if np.count_nonzero(lane_mask) > 0:
        # Find connected components in the lane mask (to separate distinct lane line segments)
        # Using 8-connectivity for pixel connection
        num_labels, labels_mat = cv2.connectedComponents(lane_mask, connectivity=8)
        # If num_labels <= 1, it means no lane pixels (only background). Otherwise, labels 1..num_labels-1 are foreground.
        if num_labels > 1:
            # Determine left and right lane clusters by their position
            frame_center_x = w / 2
            left_candidates = []   # (label, bottom_x) for labels on left side
            right_candidates = []  # (label, bottom_x) for labels on right side

            for label in range(1, num_labels):  # skip background label 0
                mask_coords = np.column_stack(np.where(labels_mat == label))
                if mask_coords.size == 0:
                    continue
                # Determine the bottom-most point's x coordinate for this cluster (highest y value = bottom of image)
                bottom_idx = np.argmax(mask_coords[:, 0])  # index of max y (row) in the coords
                bottom_y = mask_coords[bottom_idx, 0]
                bottom_x = mask_coords[bottom_idx, 1]
                # Alternatively, use mean x or median x to represent horizontal position:
                # bottom_x = np.median(mask_coords[:, 1])
                if bottom_x < frame_center_x:
                    left_candidates.append((label, bottom_x, bottom_y))
                else:
                    right_candidates.append((label, bottom_x, bottom_y))
            # Choose the label closest to center on each side (ego lane boundaries)
            left_label = None
            right_label = None
            if left_candidates:
                # Pick the candidate with largest bottom_x (closest to center from left side)
                left_label = max(left_candidates, key=lambda x: x[1])[0]
            if right_candidates:
                # Pick the candidate with smallest bottom_x (closest to center from right side)
                right_label = min(right_candidates, key=lambda x: x[1])[0]

            # Fit quadratic polynomial (x = a*y^2 + b*y + c) for each lane line if present
            def fit_poly_line(label_id):
                pts = np.column_stack(np.where(labels_mat == label_id))
                # pts columns: [y, x], meaning row index and col index of each pixel
                if pts.shape[0] < 5:
                    return None  # not enough points to fit a stable curve
                ys = pts[:, 0].astype(np.float32)
                xs = pts[:, 1].astype(np.float32)
                # Fit a second-degree polynomial for x as function of y
                poly_coeff = np.polyfit(ys, xs, 2)  # returns [a, b, c] for x = a*y^2 + b*y + c
                return poly_coeff

            if left_label is not None:
                left_line_poly = fit_poly_line(left_label)
            if right_label is not None:
                right_line_poly = fit_poly_line(right_label)

    # 6. Temporal smoothing using history buffer
    if left_line_poly is not None:
        left_history.append(left_line_poly)
    elif len(left_history) > 0:
        # If current frame has no left line, reuse last known (do not append a new one)
        pass  # (the last values in left_history will be used for smoothing)
    if right_line_poly is not None:
        right_history.append(right_line_poly)
    elif len(right_history) > 0:
        # If current frame has no right line, reuse last known
        pass

    # Compute averaged polynomial coefficients if history has entries
    if len(left_history) > 0:
        # Average coefficients across history (element-wise)
        avg_left_coeff = np.mean(left_history, axis=0)
    else:
        avg_left_coeff = None
    if len(right_history) > 0:
        avg_right_coeff = np.mean(right_history, axis=0)
    else:
        avg_right_coeff = None

    # 7. Construct lane polygon (ego lane area) if both sides are available
    if avg_left_coeff is not None and avg_right_coeff is not None:
        # Determine the vertical range for the polygon (common range where both lines exist)
        # We'll use the full frame height, but ensure to stop where one line might not be visible
        # Estimate y range where lines are valid: from bottom (h) up to a top cutoff
        # We can use the highest y value (smallest in value) of each detected line cluster as limit.
        # For simplicity, use full frame height and rely on depth filtering above to remove far points.
        y_bottom = h - 1
        y_top = 0
        # If we tracked the top of each line cluster earlier, we could use that. Here, we'll just start from bottom.
        # Sampling the polynomial curves for both lines
        lane_points_left = []
        lane_points_right = []
        step = max(1, h // 50)  # choose a step for sampling (e.g., 50 points along the height)
        for y in range(y_bottom, y_top - 1, -step):
            # Calculate x from polynomial x = a*y^2 + b*y + c
            if avg_left_coeff is not None:
                a, b, c = avg_left_coeff
                x_left = a*(y**2) + b*y + c
            else:
                continue
            if avg_right_coeff is not None:
                A, B, C = avg_right_coeff
                x_right = A*(y**2) + B*y + C
            else:
                continue
            # Ensure x_left is to the left of x_right; if not, break (lines converged or crossed)
            if x_left >= x_right:
                continue  # skip this and further up, as beyond this they converge
            # Only add points if they are within frame bounds
            if 0 <= x_left < w and 0 <= x_right < w:
                lane_points_left.append((int(x_left), y))
                lane_points_right.append((int(x_right), y))
        # If we have points for both sides, form polygon
        if lane_points_left and lane_points_right:
            # lane_points_left is from bottom to top, lane_points_right is from bottom to top
            # For polygon, we need to connect them. Use left side as is, and right side in reverse (top to bottom).
            lane_points_right_rev = lane_points_right[::-1]
            poly_points = np.array(lane_points_left + lane_points_right_rev, dtype=np.int32)
            # Draw the polygon on an overlay
            overlay = orig_frame.copy()
            cv2.fillPoly(overlay, [poly_points], color=overlay_color)
            # Blend the overlay with the original frame using transparency
            frame = cv2.addWeighted(overlay, overlay_alpha, orig_frame, 1 - overlay_alpha, 0)
        else:
            # If no valid points for polygon (perhaps lines not long enough), just copy original frame
            frame = orig_frame.copy()
    else:
        # If one or both lane lines are not available, do not overlay polygon
        frame = orig_frame.copy()

    # 8. Display the frame
    cv2.imshow("Ego Lane Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    frame_count += 1

# Cleanup resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Video processing finished. Cleaned up resources.")
