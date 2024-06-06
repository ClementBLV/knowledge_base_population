#!/bin/bash

splits=(1 5 7 10 20 100)
for spl in ${splits[@]}; do
    docker build . -t split_${spl}
done

# run containers with:
# docker run -v `pwd`/volume split_1 [--no_training]
