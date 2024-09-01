#!/bin/bash

while :
do
    date
    python jack.py -g goal
    echo "Script exited, restarting"
done
