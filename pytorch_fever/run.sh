#!/bin/bash

echo "Starting server"
python pytorch_fever/server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 3`; do
    echo "Starting client $i"
    python pytorch_fever/client.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait