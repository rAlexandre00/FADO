FROM nvidia/cuda:11.1.1-base-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.8 python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2 inetutils-ping curl host mtr-tiny tcpdump iptables \
    && rm -rf /var/lib/apt/lists/*