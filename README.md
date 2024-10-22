# base_fastapi

## Add new class and migration

- Thêm model tại folder server/app/model
- Sử dụng 'alembic revision --autogenerate' để tạo file migration
- Sử dụng 'alembic upgrade head' để update database

## Add new record in init

- Thêm data khởi tạo tại file /server/app/init_db.py
- Thay đổi thông tin của super user khi khởi tạo tại /server/.env (USER_INIT, USER_PASS)
