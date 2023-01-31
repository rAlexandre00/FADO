#!/usr/bin/env bash

# Create route for clients network
while [ -z "$clients" ]
do
  clients=$(dig +short fado_clients A)
done

# Create route for server network
while [ -z "$server" ]
do
  server=$(dig +short fado_server A)
done

# Create interfaces
ip link add link eth0 name 10.1.0.1 type ipvlan mode l2
ip addr add dev 10.1.0.1 10.1.0.1/16

ip link add link eth1 name 10.2.0.1 type ipvlan mode l2
ip addr add dev 10.2.0.1 10.2.0.1/16

# Add route to networks
ip route add 10.1.0.0/16 via "$clients"
ip route add 10.2.0.0/16 via "$server"