import json
from fastapi import WebSocket
import os
import httpx

class ConnectionManager:
    def __init__(self):
        self.__active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.__active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.__active_connections.remove(websocket)
    
    async def send_response(self, response: dict, websocket: WebSocket):
        message = json.dumps(response)
        await websocket.send_text(message)

    async def broadcast(self, response: dict):
        for connection in self.__active_connections:
            await connection.send_text(json.dumps(response))