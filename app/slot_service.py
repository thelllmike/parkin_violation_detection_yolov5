from ultralytics import YOLO
import cv2
import numpy as np
import base64

class SlotDetectionService:
    def __init__(self, model_path):
        """Initialize with pre-trained parking slot detection model"""
        self.model = YOLO(model_path)
        self.class_names = {
            0: 'empty',
            1: 'occupied'
        }

    def detect_slots(self, image_bytes):
        """Detect parking slots using pre-trained model"""
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")

        # Run inference with pre-trained model
        results = self.model.predict(
            source=image,
            conf=0.25,  # Confidence threshold
            iou=0.45    # NMS IOU threshold
        )
        detections = results[0]

        # Create annotated image
        annotated_img = image.copy()

        # Process detections
        slots = []
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            status = self.class_names[cls_id]
            color = (0, 255, 0) if status == 'empty' else (0, 0, 255)  # Green for empty, Red for occupied
            
            # Draw box and label
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                f"{status} ({conf:.2f})",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            slots.append({
                "status": status,
                "confidence": round(conf, 2),
                "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

        # Encode annotated image
        success, encoded_image = cv2.imencode(".png", annotated_img)
        if not success:
            raise ValueError("Failed to encode annotated image")
        
        b64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

        # Prepare results
        return {
            "slots": slots,
            "total_slots": len(slots),
            "occupied_slots": len([s for s in slots if s["status"] == "occupied"]),
            "empty_slots": len([s for s in slots if s["status"] == "empty"]),
            "annotated_image": b64_image
        }
