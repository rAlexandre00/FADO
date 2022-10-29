#!/usr/bin/env bash

### THIS FILE WILL BE REPLACED. IT'S HERE AS DOCUMENTATION ###
while [ -z "$fado_server" ]
do
fado_server=$(dig +short fado_server A)
done
while [ -z "$fado_client_1" ]
do
fado_client_1=$(dig +short fado_beg-client-1 A)
done
while [ -z "$fado_client_2" ]
do
fado_client_2=$(dig +short fado_beg-client-2 A)
done
while [ -z "$fado_client_3" ]
do
fado_client_3=$(dig +short fado_beg-client-3 A)
done

iptables -t nat -A PREROUTING -p tcp --dport 8890 -j DNAT --to-destination "$fado_server":8890
iptables -t nat -A PREROUTING -p tcp --dport 8891 -j DNAT --to-destination "$fado_client_1":8891
iptables -t nat -A PREROUTING -p tcp --dport 8892 -j DNAT --to-destination "$fado_client_2":8892
iptables -t nat -A PREROUTING -p tcp --dport 8893 -j DNAT --to-destination "$fado_client_3":8893
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE
