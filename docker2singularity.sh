#!/bin/bash
docker build --file Dockerfile --rm -t sro .
rm -f singularity_images/*img
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v /home/ian/Experiments/Self_Regulation_Ontology/singularity_images:/output --privileged -t singularityware/docker2singularity sro
echo Finished Conversion
cd singularity_images
bash transfer_image.sh
echo Finished Transfer
