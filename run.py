import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from model import SignCNN

# -----------------------------
# INIT APP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend can call backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODEL (ONLY ONCE)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignCNN().to(device)

model.load_state_dict(torch.load("signcnn.pth", map_location=device))
model.eval()

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # MUST match training

    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32)

    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    return tensor

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    tensor = preprocess(image_bytes)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    letter = chr(pred + 65)

    return {"letter": letter}

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)