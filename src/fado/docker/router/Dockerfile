FROM ralexandre00/fado-requirements:latest

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2 inetutils-ping curl host mtr-tiny tcpdump iptables net-tools dnsutils libnfnetlink-dev libnetfilter-queue-dev gcc iptables
ADD fado fado
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY run_router.sh run_router.sh
COPY apply_nat.sh apply_nat.sh
RUN chmod +x run_router.sh
RUN chmod +x apply_nat.sh

COPY fado_router.py fado_router.py