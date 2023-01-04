FROM python:3.9.10-slim
RUN apt-get update && apt-get install -y build-essential make gcc && apt-get -y install nginx

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/