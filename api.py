from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncpg
from datetime import datetime
from model import CNNClassifier
from utils import predict_audio_class
import torch

app = FastAPI()


@app.post("/upload-wav/")
async def upload_wav(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        return JSONResponse(content={"error": "파일은 WAV 형식이어야 합니다."}, status_code=400)

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    label = predict_audio_class(model, file_path)
    await save_to_db(label)

    os.remove(file_path)

    return {"predicted_label": label}


async def save_to_db(label: str):
    conn = await asyncpg.connect(
        host="0.0.0.0",
        port=5432,
        user="",
        password="",
        database="db"
    )

    timestamp = datetime.now()

    await conn.execute('''
    INSERT INTO public.status (datetime, crying) VALUES ($1, $2)
    ''', timestamp, label)

    await conn.close()


if __name__ == "__main__":
    best_model_path = './model/model_1.pth'
    label_map = {0: 'Belly_pain', 1: 'Cold_hot', 2: 'Discomfort', 3: 'Don’t_know', 4: 'Hungry', 5: 'Lonely',
                 6: 'Needs_to_burp', 7: 'Scared', 8: 'Tired'}

    input_shape = (1, 8000)
    num_label = len(label_map)
    model = CNNClassifier(input_shape, num_label)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    uvicorn.run(app, host="0.0.0.0", port=1213)
