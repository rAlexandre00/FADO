#!/bin/bash

SUBNET=1
SUBNET_TO_REACH=2

create_interface() {
	ip link add link eth0 name $1 type ipvlan mode l2
	ip addr add dev $1 $1/16
	ip link set dev $1 up
}

create_interface 10.1.0.2
ip route add 10."$SUBNET_TO_REACH".1.0/32 dev 10."$SUBNET".0.2 via 10."$SUBNET".0.1

created_interfaces=0
for device_ip in {1..254}.{0..255}
do
  # Create new network namespace
  ip netns add ns10."$SUBNET".$device_ip
  # Create new virtual interface
	ip link add link eth0 name 10."$SUBNET".$device_ip type ipvlan mode l2
	# Move the created interface to the new network namespace
	ip link set 10."$SUBNET".$device_ip netns ns10."$SUBNET".$device_ip
	# Give IP to interface
	ip netns exec ns10."$SUBNET".$device_ip ip a add dev 10."$SUBNET".$device_ip 10."$SUBNET".$device_ip/16
	# Activate loopback and new interface
	ip netns exec ns10."$SUBNET".$device_ip ip link set lo up
	ip netns exec ns10."$SUBNET".$device_ip ip link set 10."$SUBNET".$device_ip up
	# Add route to new ns
	ip netns exec ns10."$SUBNET".$device_ip ip route add 10."$SUBNET_TO_REACH".1.0/32 dev 10."$SUBNET".$device_ip via 10."$SUBNET".0.1

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