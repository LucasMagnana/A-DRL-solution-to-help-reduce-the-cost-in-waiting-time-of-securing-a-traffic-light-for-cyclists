#!/bin/bash
lambda=0.0
add=0.01
for i in {1..25}
do  
    lambda=`echo $lambda + $add | bc`
    echo $lambda
    python3 main.py --learning True --new-scenario True --poisson-lambda $lambda;
    python3 main.py --learning True --poisson-lambda $lambda;
done