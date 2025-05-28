#!/bin/bash

docker run                             \
    --rm                               \
    -v $PWD:/aa                        \
    alc:latest /bin/sh -c "            \
        mkdir -p /aa/temp;             \
        cd /aa/temp ;                  \
        . /opt/ros/melodic/setup.bash  \
        cd /aa/src;                    \
        catkin_init_workspace;         \
        cd /aa;                        \
        catkin_make;                   \
    "
FOLDER=$PWD/src/vandy_bluerov/results
if [ -d "$FOLDER" ]; then
    echo "$FOLDER exists."
else 
    mkdir $FOLDER
fi

. ${PWD}/pull_sensor_meshes.sh
. ${PWD}/pull_bluerov2_description.sh