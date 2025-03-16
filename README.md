# Parking Violation Detection

A FastAPI application that uses **YOLOv5** to detect vehicles crossing a designated horizontal line (80% of the frame’s height).  
When a bounding box crosses that line (i.e., `y2 > line_y`), it is labeled **"VIOLATION"**.

---

## Project Structure

Below is an example of how your project might be organized:

**Key Files:**
- **`yolo_detector.py`**: Contains the logic for loading the YOLOv5 model and detecting vehicles crossing the line.
- **`main.py`**: The FastAPI application exposing an endpoint (`/view_video`) for uploading a video. When a video is uploaded, a window opens on the server’s display showing the annotated frames in real time.
- **`weights/best.pt`**: Your custom-trained YOLOv5 model weights.
- **`requirements.txt`**: Lists the Python dependencies for the project.
- **`.gitignore`**: Specifies files/folders to ignore in version control (e.g., virtual environments, caches, etc.).

---

## Setup and Installation

1. **Clone or Download the Repository:**

   ```bash
   git clone <your_repository_url>
   cd test_2

python3 -m venv parking_violation_env
source parking_violation_env/bin/activate  # On Linux/macOS
# or
parking_violation_env\Scripts\activate.bat # On Windows

pip install --upgrade pip
pip install -r requirements.txt

uvicorn main:app --reload