#!/bin/bash

unset XDG_RUNTIME_DIR           # see https://github.com/jupyter/notebook/issues/1411

# Manually add miniconda to PATH. Don't know why the .basrc is not correctly sourced
# export PATH="/home/tao/${USER}/miniconda3/bin:$PATH"

#bash "${HOME}/adapt_conda.sh"
#source activate py35

# choose your own unique port between 8000 and 9999 or leave it random
#NOTEBOOKPORT=$(shuf -i 8000-9999 -n 1)
NOTEBOOKPORT=9955







WORKDIR="/home/tao/${USER}"
cd $WORKDIR

echo -e "\nStarting Jupyter Notebook on port ${NOTEBOOKPORT} on the $(hostname) server."
echo -e "\nSSH tunnel command : "
echo -e "\n==========- RUN IN YOUR COMPUTER TERMINAL -============\n"
echo -e "ssh -NfL ${NOTEBOOKPORT}:$(hostname):${NOTEBOOKPORT} ${USER}@titanic"
echo -e "\nThen you can use the url with the token given by jupyter\n"

jupyter-notebook --no-browser --port=${NOTEBOOKPORT} --ip='*' # Open for all ip address = dangerous ?

# This one does not give the right url.
# jupyter-notebook --no-browser --port=${NOTEBOOKPORT} --ip=$(hostname -i)  # Is this working when we work at home ?

exit 0
