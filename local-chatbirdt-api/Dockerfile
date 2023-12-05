FROM python:3.9-slim-buster

WORKDIR /user/src/app

COPY . /Folder
COPY './requirements.txt' .

RUN pip install -r requirements.txt

COPY . .


ENTRYPOINT ["python", "app.py"]
