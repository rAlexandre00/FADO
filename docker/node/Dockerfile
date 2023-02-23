FROM ralexandre00/fado-node-requirements:latest

WORKDIR /app

RUN mkdir -p /app/config
RUN mkdir -p /app/data
RUN mkdir -p /app/data/train
RUN mkdir -p /app/data/test
RUN mkdir -p /app/data/target_test

COPY run_dev.sh run_dev.sh
COPY setup.py setup.py
RUN chmod +x run_dev.sh
