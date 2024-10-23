from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi_sqlalchemy import db
from pydantic import EmailStr, BaseModel

from core.security import create_access_token
from schemas.base import DataResponse
from schemas.token import Token
from services.srv_user import UserService
from core.config import settings
from schemas.user import UserCreateRequest
router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post('/', response_model=DataResponse[Token])
def login_access_token(form_data: LoginRequest):
    user = UserService().authenticate(
        username=form_data.username, password=form_data.password)

    if not user:
        raise HTTPException(
            status_code=400, detail='Incorrect username or password')
    if not user.is_active:
        raise HTTPException(status_code=401, detail='Inactive user')

    user.last_login = datetime.now()
    db.session.commit()

    return DataResponse().success_response({
        'access_token': create_access_token(user_id=user.id)
    })
