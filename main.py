import os
import cv2
from schemas.user_schemas import UserAppCreate, UserAppOut
import torch
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from router.user_router import router as user_router
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import check_img_size, scale_boxes, non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode
import model.user_model  # ensures SQLAlchemy sees the User class
import model.cards_model
from database import Base, engine, SessionLocal, get_db
from crud.violation_crud import create_violation
from schemas.violation_schema import ViolationCreate
from router.user_router import router as user_router
from router.violation_router import router as violation_router
from router.payment_router import router as payment_router
from router.user_app_router import router as user_app_router

import uvicorn
from web3 import Web3

# ─── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
API_URL = os.getenv("API_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT_ADDR = os.getenv("CONTRACT_ADDRESS")
OWNER_ADDR = "0xcb25302EC65227698BE614CD22DF3219e1e85F1B"

if not all([API_URL, PRIVATE_KEY, CONTRACT_ADDR]):
    raise RuntimeError("Set API_URL, PRIVATE_KEY and CONTRACT_ADDRESS in .env")

# ─── Web3.py SETUP ─────────────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(API_URL))
OWNER_ADDRESS = w3.to_checksum_address(OWNER_ADDR)

contract_abi = [{
    "inputs": [
        {"internalType": "address", "name": "_user", "type": "address"},
        {"internalType": "string", "name": "_vehicleNumber", "type": "string"},
        {"internalType": "uint256", "name": "_violationFee", "type": "uint256"}
    ],
    "name": "addViolationFee",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
}]
contract = w3.eth.contract(
    address=w3.to_checksum_address(CONTRACT_ADDR),
    abi=contract_abi
)

# ────────────────────────────────────────────────────────────────────────────────

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(violation_router, prefix="/violations", tags=["violations"])
app.include_router(payment_router, prefix="/payments", tags=["payments"], dependencies=[Depends(get_db)])
app.include_router(user_app_router, prefix="/user_app", tags=["user_app"])

def load_model(weights_path="weights/best.pt", device=""):
    device = select_device(device)
    model = DetectMultiBackend(weights_path, device=device)
    check_img_size((640, 640), s=model.stride)
    return model, model.stride, model.names, model.pt

# Load the YOLO model once at startup
model_data = load_model(device="cpu")


@smart_inference_mode()
def detect_frame(frame, model_data, conf_thres=0.25, iou_thres=0.45, line_ratio=0.8):
    model, stride, names, pt = model_data
    imgsz = check_img_size((640, 640), s=stride)
    orig = frame.copy()

    # Preprocess for YOLO
    img = cv2.resize(frame, (imgsz[1], imgsz[0]))
    img = img.transpose(2, 0, 1)[None]
    im_tensor = torch.from_numpy(img).to(model.device).float() / 255.0

    # Inference + NMS
    pred = non_max_suppression(model(im_tensor), conf_thres, iou_thres)

    h, w = orig.shape[:2]
    line_y = int(h * line_ratio)
    annotator = Annotator(orig, line_width=3, example=str(names))
    cv2.line(orig, (0, line_y), (w, line_y), (255, 0, 0), 2)

    violations = []
    for det in pred:
        if det is None or not len(det):
            continue
        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4].clone(), orig.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = names[int(cls)]
            x1, y1, x2, y2 = map(int, xyxy)

            # any object crossing the line is a violation
            if y2 > line_y:
                label += " - VIOLATION"
                violations.append("UV 1998")  # hard-coded plate

            annotator.box_label(xyxy, label, color=colors(int(cls), True))

    return annotator.result(), violations


def video_feed_generator(video_path="video/IMG_6358.MOV"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    seen = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # rotate if your source is portrait
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        annotated, viols = detect_frame(frame, model_data)

        for plate in viols:
            if plate in seen:
                continue
            seen.add(plate)

            # 1) Insert into DB
            try:
                with SessionLocal() as db:
                    vio = ViolationCreate(
                        license_plate=plate,
                        fine_amount=10.0,
                        description="Boundary-crossing violation",
                        user_id=1
                    )
                    create_violation(db, vio)
            except Exception as e:
                print(f"[DB ERROR] {e}")

            # 2) Blockchain transaction with balance check
            try:
                balance = w3.eth.get_balance(OWNER_ADDRESS)
                balance_eth = w3.from_wei(balance, "ether")
                print(f"[CHAIN] Balance of {OWNER_ADDRESS}: {balance_eth} ETH")

                tx = contract.functions.addViolationFee(
                    OWNER_ADDRESS,
                    plate,
                    10
                ).build_transaction({
                    "from": OWNER_ADDRESS,
                    "nonce": w3.eth.get_transaction_count(OWNER_ADDRESS),
                    "gas": 200_000,
                    "gasPrice": w3.to_wei("20", "gwei")
                })

                # Check if balance is enough for gas
                estimated_fee = tx["gas"] * tx["gasPrice"]
                if balance < estimated_fee:
                    print("[CHAIN ERROR] Insufficient balance to cover gas fees. Please fund the wallet.")
                    continue

                signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                print(f"[CHAIN] addViolationFee → {tx_hash.hex()}")
            except Exception as e:
                print(f"[CHAIN ERROR] {e}")

        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            break

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )

    cap.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        video_feed_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Aliases for backwards compatibility
UserCreate = UserAppCreate
UserOut = UserAppOut