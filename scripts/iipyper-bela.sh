#!/bin/bash

# This script will create a new tmux session and run a Python script in one pane and a shell script in another pane.
# The idea is to use this script to run a Python script that uses iipyper to control a Bela program.
# The script can accept command line arguments, or it can load them from a .config file.
# Example:
# sh scripts/iipyper-bela.sh examples/iipyper/bela/pd/pd-osc-receive/server.py "" /path/to/bela/scripts/build_project.sh /path/to/bela/examples/PureData/osc-receive iil-python-tools

# TODO: Make arguments named instead of positional
# TODO: Check if paths are valid
# TODO: Update config file with any arguments that were provided
# TODO: Add --help option
# TODO: Add distcc support

# Arguments
PYTHON_SCRIPT_PATH=$1
PYTHON_SCRIPT_ARGS=$2
SHELL_SCRIPT_PATH=$3
SHELL_SCRIPT_ARGS=$4
CONDA_ENV=$5

# If no arguments are provided for the shell script, load them from the .config file
if [ -z "$SHELL_SCRIPT_ARGS" ]
then
    if [ -f scripts/iipyper-bela.config ]
    then
        source scripts/iipyper-bela.config
    else
        echo "Exiting: no command line arguments given, and no iipyper-bela.config file found."
        echo "(This script assumes you are running from within the iil-python-tools repo)"
        exit 1
    fi
fi

# tmux session
tmux new-session -d -s iipyper_bela_session # Create a new tmux session
tmux split-window -h # Split the window vertically

# iipyper pane
tmux select-pane -t 0 # Select pane 0 (the left pane)
tmux send-keys "conda activate $CONDA_ENV && python $PYTHON_SCRIPT_PATH $PYTHON_SCRIPT_ARGS" C-m

# bela pane
tmux select-pane -t 1
tmux send-keys "sh $SHELL_SCRIPT_PATH $SHELL_SCRIPT_ARGS" C-m

tmux attach-session -t iipyper_bela_session # Attach to the tmux session
