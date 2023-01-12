#!/usr/bin/env bash

create_interface() {
	ip link add link eth0 name $1 type ipvlan mode l2
	ip addr add dev $1 $1/16
}

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
create_interface 10.1.0.1
create_interface 10.2.0.1

# Add route to networks
ip route add 10.1.0.0/16 via "$clients"
ip route add 10.2.0.0/16 via "$server"
