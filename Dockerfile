FROM python:3.10
USER root
RUN mkdir /app
COPY . /app/

WORKDIR /app/

RUN apt-get update && pip install -r requirements.txt

CMD ["python3", "app.py"]