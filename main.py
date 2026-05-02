from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import shutil
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.6:3000",
        "http://localhost:5174",
        ""
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("small")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/api")
async def health_check():
    return {"message": "running the system"}

##### simple transcription API 
@app.post("/transcribe")
async def transribe(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = model.transcribe(input_path)
        os.remove(input_path)
        
        return {
            "success": True,
            "text": result["text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))