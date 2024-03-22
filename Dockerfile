FROM python:3.8-slim-buster
USER root
# RUN mkdir /app
# COPY . /app/
# WORKDIR /app/
# RUN pip3 install -r requirements_dev.txt
RUN pip3 install apache-airflow

ENV AIRFLOW_HOME = "/app/airflow"
# ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT = 1000
# ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING = True
RUN airflow initdb
RUN airflow users create -e tunsingh1@gmail.com -f aadarsh -l singh -p admin -r Admin -u admin
# RUN chmod 777 start.sh
# RUN apt update -y
# ENTRYPOINT [ "/bin/bash" ]
# CMD ["start.sh"]