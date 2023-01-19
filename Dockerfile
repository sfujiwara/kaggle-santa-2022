FROM python:3.9

RUN apt update
RUN apt install -y wget cmake

WORKDIR /home/workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY tasks tasks
COPY data data

RUN inv lkh-build

ENTRYPOINT [""]
