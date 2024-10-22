import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv('PROJECT_NAME', 'FASTAPI BASE')
    SECRET_KEY: str = os.getenv('SECRET_KEY', '')
    API_PREFIX: str = ''
    BACKEND_CORS_ORIGINS: list = ['*']
    DATABASE_URL: str = os.getenv('DATABASE_URL', '')
    ACCESS_TOKEN_EXPIRE_SECONDS: int = 60 * \
        60 * 24 * 7  # Token expired after 7 days
    SECURITY_ALGORITHM: str = 'HS256'
    LOGGING_CONFIG_FILE: str = os.path.join(BASE_DIR, 'logging.ini')

    USER_INIT: str = os.getenv('USER_INIT', 'admin')
    USER_PASS: str = os.getenv('USER_PASS', 'admin')
    USER_EMAIL: str = os.getenv('USER_EMAIL', 'admin@gmail.com')


settings = Settings()
