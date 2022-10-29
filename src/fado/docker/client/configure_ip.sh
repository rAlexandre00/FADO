#!/usr/bin/env bash
while [ -z "$router" ]
do
router=$(dig +short fado_router A)
done

ip route del default
ip route add default via "$router"
