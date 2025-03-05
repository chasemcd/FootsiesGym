#!/bin/bash

# Check if both arguments are provided
# First argument is training, second is eval
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <number_of_servers_from_50051> <number_of_servers_from_40051>"
  exit 1
fi

NUM_SERVERS_50051=$1
NUM_SERVERS_40051=$2

START_PORT_50051=50051
START_PORT_40051=40051

# Loop to start the specified number of servers starting from 50051
for (( i=0; i<NUM_SERVERS_50051; i++ ))
do
  PORT=$((START_PORT_50051 + i))
  echo "Starting server on port $PORT"
  ~/footsies_binaries/footsies.x86_64 --port $PORT &
done

# Loop to start the specified number of servers starting from 40051
for (( i=0; i<NUM_SERVERS_40051; i++ ))
do
  PORT=$((START_PORT_40051 + i))
  echo "Starting server on port $PORT"
  ~/footsies_binaries/footsies.x86_64 --port $PORT &
done

echo "All servers started."
