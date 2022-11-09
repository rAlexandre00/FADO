FROM ralexandre00/fado-router-base:latest

ADD fado fado
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY apply_nat.sh apply_nat.sh
RUN chmod +x apply_nat.sh