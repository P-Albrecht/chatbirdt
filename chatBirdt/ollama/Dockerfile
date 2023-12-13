FROM python:3.9-slim-buster


RUN yum install -y wget
RUN curl -X POST http://ollama:11430/api/pull -d '{"name": "llama2"}'