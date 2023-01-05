#!/bin/bash
lambda=0.0
add=0.1
mgs=3
addint=2
for i in {1..10}
do
    #lambda=`echo $lambda + $add | bc`
    mgs=`echo $mgs + $addint | bc`
    echo $mgs
    for j in {1..1}
    do
        python3 main.py --learning True --min-group-size $mgs --config 3 --struct-open True
        #python3 main.py --learning True --poisson-lambda $lambda --config 1
    done
done
