#!/bin/bash

echo "Custom setup"

dir=$(cd $(dirname "$0"); pwd)

if [ -f "${dir}/requirements.txt" ]; then
    pip3 install -r "${dir}/requirements.txt"
fi
