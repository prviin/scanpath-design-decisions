#!/usr/bin/bash
parent_folder="statistical_test"
mkdir -p $parent_folder
sub_folder="input_size"
mkdir -p $parent_folder/$sub_folder
python Statistical_test.py -i ../UEyes_dataset/images \
  -c ../UEyes_dataset/eyetracker_logs \
  -n 10  -r 0.2 \
  --width 64,128,225,512 --height 64,128,225,512 \
  --device cuda:1 --gamma 0.1 \
  --output $parent_folder/$sub_folder \
  --images-category ../UEyes_dataset/images_types_test.csv

sub_folder="radius"
mkdir -p $parent_folder/$sub_folder
for radius in 0.05 0.1 0.2 0.4; do
  mkdir -p $parent_folder/$sub_folder/$radius
  python Statistical_test.py -i ../UEyes_dataset/images \
    -c ../UEyes_dataset/eyetracker_logs \
    -n 10 \
    --width 225 --height 225 \
    --device cuda:1 --gamma 0.1 \
    --output $parent_folder/$sub_folder/$radius \
    --images-category ../UEyes_dataset/images_types_test.csv \
    --radius $radius
done

sub_folder="gamma"
mkdir -p $parent_folder/$sub_folder
for gamma in 0.1 0.5 0.9; do
  mkdir -p $parent_folder/$sub_folder/$gamma
  python Statistical_test.py -i ../UEyes_dataset/images \
    -c ../UEyes_dataset/eyetracker_logs \
    -n 10  -r 0.2 \
    --width 225 --height 225 \
    --device cuda:1 --gamma $gamma \
    --output $parent_folder/$sub_folder/$gamma \
    --images-category ../UEyes_dataset/images_types_test.csv
done

sub_folder="n"
mkdir -p $parent_folder/$sub_folder
for n in 5 7 10; do
  mkdir -p $parent_folder/$sub_folder/$n
  python Statistical_test.py -i ../UEyes_dataset/images \
    -c ../UEyes_dataset/eyetracker_logs \
    -n $n  -r 0.2 \
    --width 225 --height 225 \
    --device cuda:1 --gamma 0.1 \
    --output $parent_folder/$sub_folder/$n \
    --images-category ../UEyes_dataset/images_types_test.csv
done

