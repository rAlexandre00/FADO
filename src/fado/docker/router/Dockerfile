FROM ubuntu:bionic

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2 inetutils-ping curl host mtr-tiny tcpdump iptables net-tools dnsutils bind9 bind9utils bind9-doc 
    
# Enable IPv4
RUN sed -i 's/OPTIONS=.*/OPTIONS="-4 -u bind"/' /etc/default/bind9

COPY named.conf /etc/bind/named.conf

WORKDIR /app

COPY run_router.sh run_router.sh
RUN chmod +x run_router.sh
