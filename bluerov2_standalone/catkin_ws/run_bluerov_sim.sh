#!/bin/bash

docker run -it \
           --rm \
           --name bluerov_sim \
           --entrypoint=/bin/bash \
           --network ros \
           --ip 172.18.0.4 \
           --hostname aa_uuvsim \
           --add-host $(hostname):172.18.0.1 \
           -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
           -v $PWD/results:/mnt/results \
           -v $PWD:/aa \
           -v $ALC_HOME:$ALC_HOME\
           -v $ALC_WORKING_DIR:$ALC_WORKING_DIR\
           -e DISPLAY=$DISPLAY \
           -e SDL_VIDEODRIVER=x11 \
           -e ROS_MASTER_URI=http://ros-master:11311 \
           -e ROS_HOSTNAME=aa_uuvsim \
           -e ALC_SRC=$ALC_HOME \
           -e ALC_HOME=$ALC_HOME \
           -e ALC_WORKING_DIR=$ALC_WORKING_DIR \
           -e PYTHONPATH=$PYTHONPATH:$ALC_SRC \
           -w /aa \
           alc:latest

#and then source run_xvfb.sh
