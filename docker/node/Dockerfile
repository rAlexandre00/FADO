ARG VERSION=latest
FROM ralexandre00/fado-node-requirements:${VERSION}

WORKDIR /app

RUN mkdir -p /app/config
RUN mkdir -p /app/data
RUN mkdir -p /app/data/train
RUN mkdir -p /app/data/test
RUN mkdir -p /app/data/target_test

ENV FADO_CONFIG_PATH /app/config/fado_config.yaml
ENV FADO_DATA_PATH /app/data
ENV LOG_FILE_PATH /app/logs
ENV SERVER_IP 10.0.0.2


COPY run_dev.sh run_dev.sh
COPY setup.py setup.py
RUN chmod +x run_dev.sh
