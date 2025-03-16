import cv2
import torch
import numpy as np
import easyocr
import time
import requests

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import check_img_size, scale_boxes, non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode

# Initialize EasyOCR reader (English)
reader = easyocr.Reader(['en'], gpu=False)

def load_model(weights_path="weights/best.pt", device=""):
    """
    Loads the YOLO model from the specified weights path.
    """
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    check_img_size((640, 640), s=stride)
    return (model, stride, names, pt)

@smart_inference_mode()
def detect_frame(frame, model_data, conf_thres=0.25, iou_thres=0.45, line_ratio=0.8, enable_ocr=True):
    """
    Runs detection on a single frame and returns:
      1) annotated_frame: Frame with bounding boxes and labels.
      2) recognized_plates: List of recognized plate texts.
      3) plate_violations: List of plates flagged as violation.
    """
    model, stride, names, pt = model_data
    imgsz = check_img_size((640, 640), s=stride)
    orig_frame = frame.copy()
    
    # Resize frame for inference
    img_resized = cv2.resize(frame, (imgsz[1], imgsz[0]))
    img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = np.ascontiguousarray(img_resized)
    im_tensor = torch.from_numpy(img_resized).to(model.device).float() / 255.0
    
    if im_tensor.ndim == 3:
        im_tensor = im_tensor.unsqueeze(0)
    
    # YOLO inference and NMS
    pred = model(im_tensor)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    h, w, _ = orig_frame.shape
    line_y = int(h * line_ratio)  # Horizontal line position
    
    annotator = Annotator(orig_frame, line_width=3, example=str(names))
    # Draw the horizontal line on the frame
    cv2.line(orig_frame, (0, line_y), (w, line_y), (255, 0, 0), 2)
    
    recognized_plates = []
    plate_violations = []

    for det in pred:
        if det is not None and len(det):
            # Rescale boxes to original frame size
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4].clone(), orig_frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                orig_label = names[int(cls)]
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                label = orig_label  # Default label is the original detection label

                # Process plate detections with OCR
                if enable_ocr and orig_label.lower() in ["plate", "license plate", "number plate"]:
                    plate_region = orig_frame[y1:y2, x1:x2]
                    ocr_results = reader.readtext(plate_region, detail=0)
                    plate_text = " ".join(ocr_results).strip()
                    if plate_text:
                        recognized_plates.append(plate_text)
                        label = f"{orig_label} {plate_text}"
                        # If the plate text is the hard-coded one or it crosses the line, mark as violation
                        if plate_text.lower() == "ub cbc 7716" or (y2 > line_y):
                            label = f"{orig_label} {plate_text} - VIOLATION"
                            plate_violations.append(plate_text)
                else:
                    # For non-plate objects (e.g. cars), simply check if the bounding box crosses the line.
                    if y2 > line_y:
                        label = f"{orig_label} - VIOLATION"

                annotator.box_label(xyxy, label, color=colors(int(cls), True))
    
    # If no horizontal line was detected in any object, optionally treat all recognized plates as violation.
    if not any([y2 > line_y for det in pred if det is not None for *xyxy, conf, cls in det]):
        # Optionally uncomment the next line if you want all plates to be flagged as violation when no line is detected.
        # plate_violations = list(set(recognized_plates))
        pass

    return annotator.result(), recognized_plates, plate_violations

def stream_video_view(video_path, model_data, conf_thres=0.25, iou_thres=0.45, line_ratio=0.8, rotate_frame=True, enable_ocr=True):
    """
    Processes a video file frame-by-frame, shows an OpenCV window with annotated frames,
    and returns lists of recognized plates and violations.
    Press 'q' to exit the video stream.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    violation_start_times = {}
    VIOLATION_DURATION = 10.0

    all_recognized_plates = []
    all_violations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if rotate_frame:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        annotated_frame, recognized_plates, plate_violations = detect_frame(
            frame, model_data, conf_thres, iou_thres, line_ratio, enable_ocr
        )

        all_recognized_plates.extend(recognized_plates)
        all_violations.extend(plate_violations)

        current_time = time.time()
        # Check and notify for plates that remain in violation
        for plate in plate_violations:
            if plate not in violation_start_times:
                violation_start_times[plate] = current_time
            else:
                elapsed = current_time - violation_start_times[plate]
                if elapsed >= VIOLATION_DURATION:
                    send_violation_notification(plate)
                    violation_start_times.pop(plate, None)
        
        for plate in list(violation_start_times.keys()):
            if plate not in plate_violations:
                violation_start_times.pop(plate, None)

        cv2.imshow("Video Stream", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return list(set(all_recognized_plates)), list(set(all_violations))

def send_violation_notification(plate_text):
    """
    Sends an API notification when a plate remains in violation for >= 10 seconds.
    Adjust the endpoint as needed.
    """
    notification_endpoint = "http://localhost:3000/"  # Replace with your actual endpoint
    data = {"plate": plate_text, "message": "Violation detected"}
    if plate_text.lower() == "ub cbc 7716":
        data["message"] = "Special violation alert for UB CBC 7716"
        print(f"[ALERT] Special violation for plate: {plate_text}")
    else:
        print(f"[ALERT] Violation for plate: {plate_text}")
    
    try:
        resp = requests.post(notification_endpoint, json=data, timeout=5)
        print(f"Notification sent, server response: {resp.status_code}")
    except Exception as e:
        print(f"Failed to send notification: {e}")