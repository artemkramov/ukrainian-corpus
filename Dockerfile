FROM python:3.6-slim AS compile-image

RUN apt-get update &&  apt-get -q install -y curl gcc g++ build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements1.txt .

RUN pip install --user -r requirements1.txt

FROM python:3.6-slim AS build-image

RUN pip install flask gunicorn

COPY --from=compile-image /root/.local /usr/local

COPY . /root

WORKDIR /root

CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app", "--reload", "--timeout", "240"]