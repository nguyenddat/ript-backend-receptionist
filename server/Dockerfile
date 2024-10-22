FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get -y dist-upgrade
RUN apt-get install -y gcc python3-dev musl-dev  libffi-dev netcat-traditional

RUN mkdir /server
WORKDIR /server

RUN pip install pipenv

COPY Pipfile /server/

RUN pipenv lock
RUN pipenv requirements > requirements.txt

EXPOSE 9991
RUN pip install -r requirements.txt

COPY ./prestart.sh /server/

COPY . /server/

RUN ["chmod", "+x", "/server/prestart.sh"]

ENTRYPOINT ["/server/prestart.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
