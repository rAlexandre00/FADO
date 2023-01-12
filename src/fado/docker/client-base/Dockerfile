FROM ralexandre00/fado-requirements:latest
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY run_client.sh run_client.sh
COPY run_server.sh run_server.sh
COPY torch_client.py torch_client.py
COPY torch_server.py torch_server.py
COPY server_aggregator.py server_aggregator.py
COPY client_trainer.py client_trainer.py
COPY config_interfaces.sh config_interfaces.sh
COPY run_clients.sh run_clients.sh

RUN chmod +x run_client.sh
RUN chmod +x run_server.sh
RUN chmod +x config_interfaces.sh
RUN chmod +x run_clients.sh

## Add the wait script to the image
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

ENTRYPOINT [ ]
