#!/bin/bash

# Test

python3 test.py \
--device 0  \
--batch-size 1 \
--data data/real_world.yaml \
--img-size 480 \
--name test1 \
--weights 'weights/yolov7-tiny.pt' \
# --weights 'weights/dyvir_weights/ciou_dyvir.pt' \

