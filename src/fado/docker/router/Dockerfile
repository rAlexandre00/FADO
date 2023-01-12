FROM ralexandre00/fado-router-base:latest

ADD fado fado
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY create_interfaces.sh create_interfaces.sh
RUN chmod +x create_interfaces.sh
COPY run_router.sh run_router.sh
RUN chmod +x run_router.sh