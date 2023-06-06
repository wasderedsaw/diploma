#!/usr/bin/env bash

./train_infinite_generator.py --input_size 96
./main.py --batch_size 20 --epochs 120 --class roads --input_size 96 --train

#python3 ./main.py --batch_size 20 --epochs 120 --class water --train
#python3 ./main.py --batch_size 20 --epochs 120 --class forest --train
#python3 ./main.py --batch_size 20 --epochs 120 --class sand --train
#python3 ./main.py --batch_size 20 --epochs 120 --class clouds --train
#python3 ./main.py --batch_size 20 --epochs 120 --class water --train
#python3 ./main.py --batch_size 20 --epochs 120 --class grass --train
#python3 ./main.py --batch_size 20 --epochs 120 --class ground --train
#
#python3 ./main.py --batch_size 20 --epochs 120 --class water --classic --train
#python3 ./main.py --batch_size 20 --epochs 120 --class forest --classic --train
#python3 ./main.py --batch_size 20 --epochs 120 --class sand --classic --train
#python3 ./main.py --batch_size 20 --epochs 120 --class clouds --classic --train
#python3 ./main.py --batch_size 20 --epochs 120 --class water --classic --train
#python3 ./main.py --batch_size 20 --epochs 120 --class grass --classic --train
#python3 ./main.py --batch_size 20 --epochs 120 --class ground --classic --train