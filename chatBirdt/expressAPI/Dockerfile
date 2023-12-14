FROM python:3.9-slim-buster

WORKDIR /user/src/app

COPY './requirements.txt' .

#RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "app.py"]