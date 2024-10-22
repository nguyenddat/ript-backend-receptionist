from app.core.config import settings
from app.models.user import User
from app.helper.enums import UserRole
from app.core.security import get_password_hash
from sqlalchemy import create_engine
from app.models.base_class import Base
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()
user = db.query(User).filter_by(username=settings.USER_INIT).first()
if not user:
    new_user = User(
        full_name=settings.USER_INIT,
        email=settings.USER_EMAIL,
        username=settings.USER_INIT,
        hashed_password=get_password_hash(settings.USER_PASS),
        is_active=True,
        role=UserRole.ADMIN.value,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
db.close()
