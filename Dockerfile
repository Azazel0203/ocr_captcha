FROM python:3.10-slim-buster

USER root
WORKDIR /app

COPY . .

RUN apt-get update && pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "app.py"]