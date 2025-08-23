#!/bin/bash
list="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
for i in $list; do
python train_DTU.py -s Data/DTU/scan${i} -m Output/DTU/scan${i} --depth_ratio 0 -r 2
python extract_mesh_tsdf.py -m Output/DTU/scan${i}/total
done