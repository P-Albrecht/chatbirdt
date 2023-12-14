FROM python:3.9-slim-buster
MAINTAINER Shekhar Gulati "shekhargulati84@gmail.com"

COPY . /Folder


COPY . /app
WORKDIR /app
#RUN pip install -r requirements.txt


RUN pip install --no-cache-dir flask langchain requests argparse tqdm pypdf
RUN pip install Flask-Cors
#RUN pip install -U sentence-transformers
#RUN pip install chromadb

RUN pip install pysqlite3-binary
# Install the langchain package
#RUN pip install langchain

ENTRYPOINT ["python"]
CMD ["app.py"]
