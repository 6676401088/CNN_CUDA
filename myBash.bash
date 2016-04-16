#!/bin/sh
#BSUB -J tavana
#BSUB -o output_file
#BSUB -e error_file
#BSUB -n 1
#BSUB -q par-gpu
#span[ptile=32]
#BSUB cwd /home/khavaritavana.m/CNN_GPU

work=/home/khavaritavana.m/CNN_GPU

cd $work

./cnn
./cnn
./cnn
./cnn
./cnn
./cnn
./cnn
./cnn
./cnn
./cnn

