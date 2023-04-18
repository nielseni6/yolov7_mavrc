#!/bin/bash
datasets=("data/large.yaml" "data/medium.yaml" "data/small.yaml")
epochs=(5 5 5)

for i in "${!datasets[@]}"
do
    dataset="${datasets[i]}"
    epoch="${epochs[i]}"

    if [ $i -eq 0 ]; then
        weights='weights/yolov7-tiny.pt'
    else
        weights="weights/yolov7-custom.pt"
    fi

    fname=$(echo ${datasets[i]} | sed -e 's/data\/\(.*\).yaml/\1/g')
    fname=$(echo $fname.out)

    python3 train.py \
    --epochs $epoch \
    --workers 8 \
    --device 1,2,3,4 \
    --batch-size 64 \
    --data $dataset \
    --img 480 480 \
    --entity "aslane84" \
    --name CL_yolov7_ \
    --cfg cfg/training/yolov7-tiny-drone.yaml \
    --weights $weights \
    --hyp data/hyp.drone.tiny.yaml \
    --multi-scale \
    --save_period 2 \

done
