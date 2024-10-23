import logging
import uvicorn
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware
from starlette.middleware.cors import CORSMiddleware

from api import api_login, api_user, api_faceRecognition
from models.base_class import Base
# from db.base import engine
from core.config import settings
from helper.exception_handler import CustomException, http_exception_handler

logging.config.fileConfig(settings.LOGGING_CONFIG_FILE, disable_existing_loggers=False)

# Base.metadata.create_all(bind=engine)

def get_application() -> FastAPI:
    application = FastAPI(
        # title=settings.PROJECT_NAME, docs_url="/docs", redoc_url='/re-docs',
        # openapi_url=f"{settings.API_PREFIX}/openapi.json",
        # description='Base FastAPI'
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # application.add_middleware(DBSessionMiddleware, db_url=settings.DATABASE_URL)
    application.include_router(api_user.router, tags=["user"], prefix="/users")
    application.include_router(api_login.router, tags=["login"], prefix="/login")
    application.include_router(api_faceRecognition.router, tags = ["faceRecognition"], prefix = "/faceRecognition")
    application.add_exception_handler(CustomException, http_exception_handler)
    return application


app = get_application()
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
