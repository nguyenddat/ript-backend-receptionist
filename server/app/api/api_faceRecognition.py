import os
import shutil
from typing import List, AnyStr, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from services import ConnectionManager, ImageManager, ModelManager, ExtractCCCD

router = APIRouter()
connection_manager = ConnectionManager.ConnectionManager()
image_manager = ImageManager.ImageManager()
model_manager = ModelManager.ModelManager()
model_manager._load_stored_data()
TARGET_WEBSOCKET: WebSocket = None

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global TARGET_WEBSOCKET
    TARGET_WEBSOCKET = WebSocket
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                result = model_manager.predict(data, image_manager)
                await connection_manager.send_response({
                    "success": True,
                    "event": "webcam",
                    "payload": result,
                }, websocket)
            except Exception as err:
                result = []
                await connection_manager.send_response({
                    "success": False,
                    "event": "webcam",
                    "payload": result,
                    "error": {
                        "code": status.HTTP_400_BAD_REQUEST,
                        "message": err
                    }
                }, websocket)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        TARGET_WEBSOCKET = None

@router.post("/api/get-identity")
async def get_identity(data: List[AnyStr]):
    global TARGET_WEBSOCKET, manager
    if not TARGET_WEBSOCKET:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST , detail = "Chưa có ai kết nối đến máy chủ!")
    try:
        decoded_data = ExtractCCCD.extract_data(data)
        await connection_manager.send_response({
            "success": True,
            "event": "cccd",
            "payload": decoded_data
        }, TARGET_WEBSOCKET)
    except Exception as err:
        await connection_manager.send_response({
            "success": False,
            "event": "cccd",
            "payload": {},
            "error": {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": err
            }
        })

@router.post('/api/post-personal-img')
async def post_personal_img(data: Dict[AnyStr, List[AnyStr] | Dict[AnyStr, AnyStr] | AnyStr]):
    if not data:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Không có dữ liệu"
        )

    global faces_data
    b64_img = data['b64_img']
    personal_data = data['cccd']
    role =  data['role']
    personal_id = personal_data['Identity Code']
    personal_data.update({'role': role})    

    save_img_path = os.path.join(f"./data/{personal_id}")
    if os.path.exists(save_img_path):
        shutil.rmtree(save_img_path)

    os.makedirs(save_img_path)
    try:
        id = 0 
        for img in b64_img:
            img_path = os.path.join(save_img_path, f'{personal_id}_{id}.png')
            image_manager.save_img_from_base64(img, img_path)
            print(f"Lưu thành công ảnh: {personal_id}_{id}.png")
            with open(os.path.join(save_img_path, f'{personal_id}_{id}_base64.txt'), 'w') as file:
                file.write(img)
            id += 1
        model_manager.load_new_data(save_img_path, image_manager)
        return {
            "success": True,
        }
    except Exception as err:
        shutil.rmtree(save_img_path)
        return {
            "success": False,
            "error": {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": err
            }
        }
