import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from yolo_detector import load_model, stream_video_view
import uvicorn

app = FastAPI()

# Load the YOLOv5 model once at startup. Adjust the weights path if needed.
model_data = load_model(weights_path="weights/best.pt")

@app.post("/view_video")
async def view_video(file: UploadFile = File(...)):
    """
    Endpoint to process a video file upload. The video is saved temporarily,
    then processed frame-by-frame. A separate window ("Video Stream") will open
    on the server, showing the annotated frames. Press 'q' to close.
    
    The 'rotate_frame=True' parameter rotates each frame by 90° clockwise so that
    videos recorded in portrait orientation appear horizontally.
    """
    contents = await file.read()
    # Save uploaded video to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp.flush()
        input_video_path = tmp.name

    try:
        # Call the streaming function with rotation enabled
        stream_video_view(
            video_path=input_video_path,
            model_data=model_data,
            rotate_frame=True  # <--- set to True by default
        )
    except Exception as e:
        # Cleanup on error
        os.remove(input_video_path)
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    # Cleanup after successful streaming
    os.remove(input_video_path)
    return JSONResponse(content={"message": "Video streaming finished."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)