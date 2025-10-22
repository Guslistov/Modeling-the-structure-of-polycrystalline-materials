from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import os

from fastapi import Body
#from .utils import process_eulers
from .utils import process_IPF
from .utils import process_eulers
from .utils import process_euler_reconstraction

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация путей
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "static" / "data"
MODEL_DIR = BASE_DIR / "static" / "models"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Генерируем уникальное имя файла
        file_ext = file.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = UPLOAD_DIR / file_name

        processed_filename = f"{Path(file_name).stem}.png"
        processed_full_path = UPLOAD_DIR / processed_filename
        
        # Сохраняем оригинальный файл
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        step_x = 6
        step_y = 6
        step_x, step_y = await process_eulers(file_path, processed_full_path)

        return {
            "original_file": f"/static/{file_name}",
            "original_image": f"/static/{processed_filename}",
            "step_x": float(step_x),
            "step_y": float(step_y),
            "status": "uploaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/process/")
async def process_image(data: dict = Body(...)):
    try:
        original_filename = data.get("file_name")
        if not original_filename:
            raise HTTPException(status_code=400, detail="File name required")

        original_full_path = UPLOAD_DIR / original_filename
        
        if not original_full_path.exists():
            raise HTTPException(status_code=404, detail="Original file not found")

        #filename = original_full_path.split('/')[-1].split('.')[0] + "_IPF_" + "Z" + ".png"
        #dir_path = '/'.join(original_full_path.split('/')[:-1])
        #out_path = dir_path + "/" + filename + "_IPF_" + "Z" + ".png"
        #processed_filename = f"processed_{original_filename}"
        processed_filename = f"processed_{Path(original_filename).stem}.png"
        processed_full_path = UPLOAD_DIR / processed_filename
        
        # Обрабатываем и сохраняем файл с префиксом processed_
        #await process_eulers(original_full_path, processed_full_path)
        type = data.get("IPF_type")
        await process_IPF(original_full_path, processed_full_path,type)
        
        if not processed_full_path.exists():
            raise HTTPException(status_code=500, detail="Processed image not created")
        
        return {
            "processed": f"/static/{processed_filename}",
            "status": "processed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process1/")
async def process_reconstraction(data: dict = Body(...)):
    try:
        window_padding = data.get("window_padding")
        original_filename = data.get("file_name")
        size_y = data.get("step_y")
        size_x = data.get("step_x")
        resized = data.get("resized") 
        model_name = data.get("model_name")

        if not original_filename:
            raise HTTPException(status_code=400, detail="File name required")

        original_full_path = UPLOAD_DIR / original_filename
        
        
        if not original_full_path.exists():
            raise HTTPException(status_code=404, detail="Original file not found")

        processed_filename = f"processed_{original_filename}"
        processed_full_path = UPLOAD_DIR / processed_filename

        processed_filename_image = f"processed_{Path(original_filename).stem}.png"
        processed_full_path_image = UPLOAD_DIR / processed_filename_image
        
        # Обрабатываем и сохраняем файл с префиксом processed_
        model_full_path = MODEL_DIR / model_name
        await process_euler_reconstraction(original_full_path, processed_full_path, model_full_path, window_padding, size_y, size_x, resized)

        await process_eulers(processed_full_path, processed_full_path_image)
        
        if not processed_full_path.exists():
            raise HTTPException(status_code=500, detail="Processed data not created")
        
        return {
            "processed_file": f"/static/{processed_filename}",
            "processed_image": f"/static/{processed_filename_image}",
            "status": "processed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

@app.get("/models/")
def list_models():
    try:
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
        return {"models": files}
    except Exception as e:
        return {"models": [], "error": str(e)}