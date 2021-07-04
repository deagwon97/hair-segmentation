#!bin/bash

SHELL_PATH=`pwd -P`

docker run --gpus all --ipc=host  -it -v $SHELL_PATH:/workspace/ \
                                        -v /DATA:/DATA aihair:all
