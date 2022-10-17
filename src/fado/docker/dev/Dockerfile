FROM python:3.8-slim-buster
WORKDIR /app

ADD fado fado
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY run_client.sh run_client.sh
COPY run_server.sh run_server.sh
COPY run_malicious_client.sh run_malicious_client.sh
COPY torch_client.py torch_client.py
COPY torch_server.py torch_server.py
COPY get_model.py get_model.py

## Add the wait script to the image
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

ENTRYPOINT [ ]
