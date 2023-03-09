FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.8 python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2 inetutils-ping curl host mtr-tiny tcpdump iptables dnsutils \
    && rm -rf /var/lib/apt/lists/*
    
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/"
