import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from services import WeeklySchedule

weekly_schedule = WeeklySchedule.ScheduleParser()

router = APIRouter()

@router.post("/api/post-weekly-schedule")
def post_lich_tuan(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST, 
            detail = "Không có dữ liệu"
        )
    if file.content_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raise HTTPException(status_code=400, detail = f"Supported formats: {weekly_schedule.supported_formats}")
    
    file_path = os.path.join(weekly_schedule.save_dir, "WeeklySchedule.docx")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        weekly_schedule.format_events(file_path)
        return {
            "success": True
        }
    except Exception as err:
        return {
            "success": False,
            "error": {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": err
            }
        }

@router.get("/api/get-weekly-schedule")
def get_lich_tuan():
    return weekly_schedule.get_events()

