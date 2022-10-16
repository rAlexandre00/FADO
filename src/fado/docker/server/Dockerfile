FROM python:3.8-slim-buster
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY run_server.sh run_server.sh
COPY torch_server.py torch_server.py
COPY get_model.py get_model.py

ENTRYPOINT ["/bin/bash", "run_server.sh"]
