#!/bin/bash

splits=(1 5 7 10 20 100)
for spl in ${splits[@]}; do
    echo docker build . -t split_${spl}
done
