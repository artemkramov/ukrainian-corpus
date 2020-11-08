FROM python:3.6-slim

COPY . /root

WORKDIR /root

RUN apt-get update &&  apt-get -q install -y curl gcc g++ build-essential

RUN pip install -r requirements.txt
