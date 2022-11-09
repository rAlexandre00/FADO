FROM ralexandre00/fado-client-base:latest
WORKDIR /app

ADD fado fado
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY get_model.py get_model.py

ENTRYPOINT [ ]
