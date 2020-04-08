#!/bin/bash

# Download the dataset into folder 'data'
mkdir data
cd data

wget $1
unzip spaceknow-urban-growth-test.zip

mv data imgs  # rename the 'data' folder from the archive
