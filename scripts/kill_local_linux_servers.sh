#!/bin/bash

# Get a list of all running footsies.x86_64 processes
PIDS=$(ps -xw | grep '[f]ootsies_binaries/footsies.x86_64' | awk '{print $1}')

# Check if there are any running processes
if [ -z "$PIDS" ]; then
  echo "No running footsies.x86_64 servers found."
  exit 0
fi

# Terminate each process
for PID in $PIDS
do
  echo "Terminating server with PID $PID"
  kill -9 $PID
done

echo "All servers terminated."
