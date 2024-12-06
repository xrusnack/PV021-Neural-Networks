#!/bin/bash

echo "Adding some modules"

module add java
module add maven

echo "#################"
echo "    COMPILING    "
echo "#################"

mvn clean package

echo "#################"
echo "     RUNNING     "
echo "#################"


time nice -n 19 java -cp target/pv021-1.0-SNAPSHOT.jar pv021.main.Main

python3 evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv