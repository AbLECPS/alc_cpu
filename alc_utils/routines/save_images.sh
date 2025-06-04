#!/bin/bash
. /opt/ros/melodic/setup.bash
# Check for the number of arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <config_folder> <output_folder> <file_prefix> <topic>"
    exit 1
fi

# Define variables based on command line arguments
BAG_FILE="$1/results/recording.bag"
INPUT_FOLDER="/root/.ros"
OUTPUT_FOLDER="$2"
FILE_PREFIX="$3"
TOPIC="$4"
mkdir -p $OUTPUT_FOLDER

roslaunch $ALC_HOME/alc_utils/routines/export.launch playback_rate:=5.0 bag_file:=$BAG_FILE image_topic:=$TOPIC

# Move and rename extracted images with the specified prefix
for img in $INPUT_FOLDER/frame*; do
    mv "$img" "$OUTPUT_FOLDER/${FILE_PREFIX}$(basename "$img")"
done



