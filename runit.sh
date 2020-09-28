#!/usr/bin/env bash

set -e

if [[ "$NB_USER" = "root" ]]; then
  # Remove jovyan user so that startup script does not fail
  # trying to set user id for jovyan. See here
  # https://github.com/jupyter/docker-stacks/blob/master/base-notebook/start.sh#L47
  userdel jovyan  # This still retains /home/jovyan
  echo "NB_USER is root. Deleted user 'jovyan'."
fi

# Launch normal start up script.
start-notebook.sh
