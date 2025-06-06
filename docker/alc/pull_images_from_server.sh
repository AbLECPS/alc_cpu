#!/bin/bash

server_name="ablecps"

image_list=("xpra_16.04:latest"
            "alc_data:latest"
            )

for (( i=0; i<${#image_list[*]}; ++i)); do
    image_name=${image_list[$i]}
    image_tagged_name=${server_name}/$image_name
    echo 'pulling image from server - '$image_tagged_name
    docker pull $image_tagged_name
    if [[ "$(docker images -q $image_name 2> /dev/null)" != "" ]]; then
        echo 'removing old image - '$image_name
        docker rmi $image_name
    fi
    echo 'tagging image from server for use - '$image_name
    docker tag $image_tagged_name $image_name
done


server_name="git.isis.vanderbilt.edu:5050/alcdockers/alc"

image_list=("alc:cpu"
            )

for (( i=0; i<${#image_list[*]}; ++i)); do
    image_name=${image_list[$i]}
    image_tagged_name=${server_name}/$image_name
    echo 'pulling image from server - '$image_tagged_name
    docker pull $image_tagged_name
    if [[ "$(docker images -q $image_name 2> /dev/null)" != "" ]]; then
        echo 'removing old image - '$image_name
        docker rmi $image_name
    fi
    echo 'tagging image from server for use - '$image_name
    docker tag $image_tagged_name $image_name
done

image_list=("jkarlos/git-server-docker:latest"
            "erichough/nfs-server:latest"
            "mongo:3.4.6"
            "nginx:stable"
            "ros:kinetic-ros-core")

for (( i=0; i<${#image_list[*]}; ++i)); do
    image_name=${image_list[$i]}
    echo 'pulling image from server - '$image_name
    docker pull $image_name
done

docker tag alc:cpu alc:latest
docker tag alc:cpu alc_alc:latest
