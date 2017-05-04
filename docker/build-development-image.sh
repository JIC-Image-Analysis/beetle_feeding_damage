#!/bin/bash

IMAGE_NAME="beetle_feeding_damage-development"

cp ../requirements.txt $IMAGE_NAME
sleep 1
cd $IMAGE_NAME
docker build --no-cache -t $IMAGE_NAME .
docker run --rm $IMAGE_NAME pip freeze > requirements.txt
mv requirements.txt ../..
cd ../
