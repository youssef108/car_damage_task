
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import base64
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import Dict, Tuple
import uuid
MODEL_PATHS = {
    "Large_without_aug": r"models\Large_no_aug.pt",
    "Large_aug":r"models\Large_with_aug.pt",
    "medium_no_aug":r"models\medium_no_aug.pt",
    "medium_with_aug":r"models\medium_with_aug.pt"
}
IMAGEDIR = r"assets/"
MODELS = {}  # Will store loaded YOLO models

app = FastAPI(title="Multi-Model YOLO Inference API")

@app.on_event("startup")
async def startup():
    """Load YOLO models once at startup."""
    for model_name, model_path in MODEL_PATHS.items():
        try:
            model = YOLO(model_path)
            MODELS[model_name] = model
            print(f"[INFO] Loaded model '{model_name}' from {model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load model '{model_name}': {e}")

def run_inference(model: YOLO, image:str) -> Tuple[Dict[str, int], str]:
    """Run inference and return class_counts + annotated image (base64)."""
    results_list = model(image)
    if not results_list:
        raise ValueError("No results from YOLO model.")

    result = results_list[0]
    boxes = result.boxes

    class_counts = {}
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names.get(class_id, str(class_id))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    annotated_array = result.plot()  # (H, W, 3) BGR
    annotated_array = annotated_array[:, :, ::-1]  # convert BGR to RGB
    pil_img = Image.fromarray(annotated_array.astype(np.uint8))

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return class_counts, encoded_img


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
 
    #save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
 
   
    if model_name not in MODELS:
        valid_names = list(MODELS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {valid_names}"
        )

    try:
        class_counts, annotated_b64 = run_inference(MODELS[model_name], f"{IMAGEDIR}{file.filename}")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # 6) Return results
    return {
        "class_counts": class_counts,
        "annotated_image_base64": annotated_b64
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Model YOLO Inference API!"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
