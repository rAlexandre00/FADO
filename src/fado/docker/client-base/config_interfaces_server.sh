#!/bin/bash

SUBNET=2
SUBNET_TO_REACH=1

create_interface() {
	ip link add link eth0 name $1 type ipvlan mode l2
	ip addr add dev $1 $1/16
	ip link set dev $1 up
}

created_interfaces=0
for device_ip in {1..254}.{0..255}
do
	create_interface 10."$SUBNET".$device_ip
	((created_interfaces++==0))
	if [[ $created_interfaces -eq N_INTERFACES ]]; then
    break
  fi
done

ip route add 10.1.0.0/16 dev 10.2.1.0 via 10.2.0.1

# Create route for server network
while [ -z "$router" ]
do
  router=$(dig +short fado_router A)
done