FROM ralexandre00/fado-node-requirements:latest

WORKDIR /app

RUN mkdir -p /app/config

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY run_dev.sh run_dev.sh
RUN chmod +x run_dev.sh

RUN apt-get update -y
RUN apt-get install -y libnetfilter-queue-dev
RUN pip3 install -r requirements.txt
