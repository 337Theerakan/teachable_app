from fastapi import APIRouter, File, UploadFile, HTTPException
from app.core.model import model_instance
import os
import uuid

router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # บันทึกไฟล์ชั่วคราว
    file_ext = os.path.splitext(file.filename)[1]
    temp_file = f"/tmp/{uuid.uuid4()}{file_ext}"
    
    try:
        # บันทึกไฟล์อัปโหลด
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        
        # ทำนายภาพ
        result, error = model_instance.predict_image(temp_file)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # ลบไฟล์ชั่วคราว
        if os.path.exists(temp_file):
            os.remove(temp_file)