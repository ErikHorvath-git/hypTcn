#!/usr/bin/env bash
iptables -t nat -A POSTROUTING -s 192.168.122.0/24 ! -d 192.168.122.0/24 -j MASQUERADE
iptables -I FORWARD -i virbr0 -j ACCEPT
iptables -I FORWARD -o virbr0 -j ACCEPT
echo "NAT OK"
