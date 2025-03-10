import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import check_img_size, scale_boxes, non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode

def load_model(weights_path="models/yolov5s.pt", device=""):
    """
    Loads the YOLOv5 model from the specified weights path.
    """
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    # Ensure the inference image size is compatible with the model stride
    check_img_size((640, 640), s=stride)
    return (model, stride, names, pt)

@smart_inference_mode()
def detect_frame(frame, model_data, conf_thres=0.25, iou_thres=0.45, line_ratio=0.8):
    """
    Runs detection on a single frame and returns an annotated frame with bounding boxes.
    
    Args:
        frame (numpy.ndarray): BGR image (frame).
        model_data (tuple): (model, stride, names, pt) from load_model().
        conf_thres (float): Confidence threshold.
        iou_thres (float): NMS IOU threshold.
        line_ratio (float): Fraction of the image height for the horizontal line.

    Returns:
        annotated_frame (numpy.ndarray): The frame with bounding boxes and a horizontal line.
    """
    model, stride, names, pt = model_data
    imgsz = check_img_size((640, 640), s=stride)
    orig_frame = frame.copy()
    
    # Resize frame to inference size and prepare tensor
    img_resized = cv2.resize(frame, (imgsz[1], imgsz[0]))
    img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = np.ascontiguousarray(img_resized)
    im_tensor = torch.from_numpy(img_resized).to(model.device).float() / 255.0
    
    if im_tensor.ndim == 3:
        im_tensor = im_tensor.unsqueeze(0)
    
    # Inference
    pred = model(im_tensor)
    # Non-max suppression
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # Horizontal line (80% of the frame’s height)
    h, w, _ = orig_frame.shape
    line_y = int(h * line_ratio)
    
    # Annotator for drawing
    annotator = Annotator(orig_frame, line_width=3, example=str(names))
    # Draw the horizontal line
    cv2.line(orig_frame, (0, line_y), (w, line_y), (255, 0, 0), 2)
    
    # Process detections
    for det in pred:
        if det is not None and len(det):
            # Rescale boxes to the original frame size
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4].clone(), orig_frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                label = names[int(cls)]
                # If bounding box crosses the line, label as "VIOLATION"
                if y2 > line_y:
                    label = "VIOLATION"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
    
    return annotator.result()

def stream_video_view(
    video_path,
    model_data,
    conf_thres=0.25,
    iou_thres=0.45,
    line_ratio=0.8,
    rotate_frame=True
):
    """
    Processes a video file frame-by-frame, applies detection, and displays the annotated frames
    in a separate window (like the original YOLOv5 script). Press 'q' to exit.
    
    Args:
        video_path (str): Path to the input video file.
        model_data (tuple): (model, stride, names, pt) from load_model().
        conf_thres (float): Confidence threshold.
        iou_thres (float): NMS IOU threshold.
        line_ratio (float): Fraction of the image height for the horizontal line.
        rotate_frame (bool): Whether to rotate each frame by 90 degrees clockwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate the frame if needed
        if rotate_frame:
            # Rotate 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Detect and annotate the frame
        annotated_frame = detect_frame(frame, model_data, conf_thres, iou_thres, line_ratio)
        
        # Show the frame in a window named "Video Stream"
        cv2.imshow("Video Stream", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()