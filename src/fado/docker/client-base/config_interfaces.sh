#!/bin/bash

SUBNET=$1
SUBNET_TO_REACH=$2

create_interface() {
	ip link add link eth0 name $1 type ipvlan mode l2
	ip addr add dev $1 $1/16
}

created_interfaces=0
for ip in 10."$SUBNET".{1..254}.{0..255}
do
	create_interface $ip
	((created_interfaces++==0))
	if [[ $created_interfaces -eq N_INTERFACES ]]; then
    break
  fi
done

# Create route for server network
while [ -z "$router" ]
do
  router=$(dig +short fado_router A)
done

# Then add route to that network
ip route add 10."$SUBNET_TO_REACH".0.0/16 via "$router"