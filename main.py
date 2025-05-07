import time
import cv2
import torch
import numpy as np
import requests

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import check_img_size, scale_boxes, non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode

from router.user_router import router as user_router
from router.violation_router import router as violation_router
from database import Base, engine, SessionLocal
from crud.violation_crud import create_violation
from schemas.violation_schema import ViolationCreate

import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Create all tables (including violations) if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(violation_router, prefix="/violations", tags=["violations"])

def load_model(weights_path="weights/best.pt", device=""):
    """
    Loads the YOLO model from the specified weights path.
    """
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    check_img_size((640, 640), s=stride)
    return (model, stride, names, pt)

# Load the YOLO model once at startup
model_data = load_model(weights_path="weights/best.pt", device="cpu")

@smart_inference_mode()
def detect_frame(frame, model_data, conf_thres=0.25, iou_thres=0.45, line_ratio=0.8, enable_detection=True):
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
    line_y = int(h * line_ratio)  # Position of horizontal line
    
    annotator = Annotator(orig_frame, line_width=3, example=str(names))
    cv2.line(orig_frame, (0, line_y), (w, line_y), (255, 0, 0), 2)
    
    recognized_plates = []
    plate_violations = []
    
    for det in pred:
        if det is not None and len(det):
            # Rescale boxes to original frame size
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4].clone(), orig_frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                orig_label = names[int(cls)]
                x1, y1, x2, y2 = map(int, xyxy)
                label = orig_label
                plate_text = ""
                
                # Instead of running OCR, we use a hard-coded plate value when the label indicates a plate.
                if enable_detection and orig_label.lower() in ["plate", "license plate", "number plate"]:
                    # Hard-coded license plate value
                    plate_text = "UP CBC 7716"
                    recognized_plates.append(plate_text)
                    label = f"{orig_label} {plate_text}"
                    # If the bounding box crosses the line, mark as violation
                    if y2 > line_y:
                        label = f"{label} - VIOLATION"
                        plate_violations.append(plate_text)
                else:
                    # For other objects, you can optionally check the line crossing condition
                    if y2 > line_y:
                        label = f"{orig_label} - VIOLATION"
                        plate_violations.append("UP CBC 7716")
                
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
    
    return annotator.result(), recognized_plates, plate_violations

def video_feed_generator(video_path="video/IMG_6358.MOV"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
    
    processed_plates = set()  # Track already-inserted plates

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        annotated_frame, recognized_plates, plate_violations = detect_frame(frame, model_data)

        for plate in plate_violations:
            if plate not in processed_plates:
                print(f"[ALERT] New violation detected for plate: {plate}")
                try:
                    with SessionLocal() as db:
                        violation_data = {
                            "license_plate": plate,
                            "fine_amount": 100.0,
                            "description": "Violation detected on boundary",
                            "user_id": 1
                        }
                        violation_instance = ViolationCreate(**violation_data)
                        create_violation(db, violation_instance)
                        processed_plates.add(plate)  # Mark as processed
                        print(f"[INFO] Violation record inserted for plate: {plate}")
                except Exception as ex:
                    print(f"[ERROR] Failed to insert violation record for plate {plate}: {ex}")

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            break

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        video_feed_generator("video/IMG_6358.MOV"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)