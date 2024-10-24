from sqlalchemy import Column, String, Boolean, DateTime, Integer, Sequence

from models.base_class import BareBaseModel


class User(BareBaseModel):
    full_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    role = Column(String, default='guest')
    last_login = Column(DateTime)
