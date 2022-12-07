#!/bin/bash
lambda=0.0
add=0.1
for i in {1..10}
do
    lambda=`echo $lambda + $add | bc`
    echo $lambda
    for j in {1..3}
    do
        python3 main.py --learning True --new-scenario True --poisson-lambda $lambda --config 0
        #python3 main.py --learning True --poisson-lambda $lambda;
    done
done
