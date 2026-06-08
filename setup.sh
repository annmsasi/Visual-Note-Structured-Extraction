#!/bin/bash

# Finds the path to python3 on this computer
PYTHON_PATH=$(which python3)

# Finds the path to main.py based on where this script lives
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/run_full_pipeline_.py"

# Writes the cron line into crontab.txt
echo "0 9 * * * $PYTHON_PATH $SCRIPT_PATH" > crontab.txt

# Loads it into the system
crontab crontab.txt

echo "Cron job set up successfully!"
echo "It will run: $PYTHON_PATH $SCRIPT_PATH"