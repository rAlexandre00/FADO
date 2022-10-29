FROM ubuntu:bionic

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2 inetutils-ping curl host mtr-tiny tcpdump iptables net-tools dnsutils

WORKDIR /app

COPY run_router.sh run_router.sh
COPY apply_nat.sh apply_nat.sh
RUN chmod +x run_router.sh
RUN chmod +x apply_nat.sh