#!/bin/bash
## change this file to your needs

echo "Adding some modules"

# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

mvn clean package

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 java -cp target/pv021-1.0-SNAPSHOT.jar pv021.main.Main

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network

python3 evaluator/evaluate.py predictions.csv data/fashion_mnist_test_labels.csv