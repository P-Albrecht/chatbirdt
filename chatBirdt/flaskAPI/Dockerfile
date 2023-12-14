FROM python:3.9-slim-buster

WORKDIR /user/src/app

COPY './requirements.txt' .

#RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pysqlite3-binary

COPY . .

ENTRYPOINT ["python", "app.py"]