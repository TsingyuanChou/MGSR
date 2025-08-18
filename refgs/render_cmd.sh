#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python render.py -s data/art1 --eval -m output/art1_ablation_ref0.00001 --render_images --skip_train --iteration 30000 > ablation_render_art1 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python render.py -s data/art2 --eval -m output/art2_ablation_ref0.00001 --render_images --skip_train --iteration 30000 > ablation_render_art2 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python render.py -s data/art3 --eval -m output/art3_ablation_ref0.00001 --render_images --skip_train --iteration 30000 > ablation_render_art3 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python render.py -s data/mirror --eval -m output/mirror_ablation_ref0.00001 --render_images --skip_train --iteration 30000 > ablation_render_mirror 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python render.py -s data/tv --eval -m output/tv_ablation_ref0.00001 --render_images --skip_train --iteration 30000 > ablation_render_tv 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python render.py -s data/bookcase --eval -m output/bookcase_ablation_ref0.00001 --render_images --skip_train --iteration 30000 > ablation_render_bookcase 2>&1 &