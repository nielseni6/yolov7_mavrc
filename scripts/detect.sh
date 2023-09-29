#!/bin/bash

# Test
python3 detect.py \
--device cpu \
--weights 'weights/best.pt' \
--conf 0.50 \
--img-size 480 \
--no-trace \
--name "practice_data" \
--cam "grad" \
--source /data/Koutsoubn8/ijcnn_v7data/Real_world_test/images \
# --source /home/nielseni6/PythonScripts/yolov7_mavrc/practice_data/small_set/yolo/images \
# --source /home/koutsoubn8/yolov7_mavrc/eigen_drone.png \
# --source /home/koutsoubn8/yolov7_mavrc/vids/Fail.mp4 \
# --source /home/koutsoubn8/yolov7_mavrc/tchparkdronepic.png \
# --source /home/koutsoubn8/yolov7_mavrc/Fail.mov \
# --source /home/koutsoubn8/yolov7_mavrc/testvid.mp4
# --source /home/koutsoubn8/yolov7_mavrc/techpark_drone.mp4 \
#for cam use --cam --no-trace and cpu if it crashes
# --weights 'weights/dyvir_weights/nwd_dyvir_ft.pt' \
# --cam "eigen" \
# --cam "grad" \