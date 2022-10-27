#!/usr/bin/env bash
router=$(dig +short fado_router)
ip route del default
ip route add default via "$router"

printf "nameserver ${router}" > /etc/resolv.conf
