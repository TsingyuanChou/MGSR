from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from geo_utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from geo_utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    with open("results.txt", 'a') as result_file:
        for scene_dir in model_paths:
            try:
                scene_name = Path(os.path.dirname(scene_dir)).name  
                print("Scene:", scene_name)
                full_dict[scene_dir] = {}
                per_view_dict[scene_dir] = {}
                
                test_dir = Path(scene_dir) / "test"
                for method in os.listdir(test_dir):
                    print("Method:", method)
                    method_dir = test_dir / method
                    gt_dir = method_dir / "gt"
                    renders_dir = method_dir / "renders"
                    renders, gts, image_names =  readImages(renders_dir, gt_dir)

                    ssims, psnrs = [], []
                    for idx in tqdm(range(len(renders)), desc="Desc:"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))

                    mean_ssim = torch.tensor(ssims).mean().item()
                    mean_psnr = torch.tensor(psnrs).mean().item()
                    #print(mean_ssim,mean_psnr)
                    result_file.write(f"Scene: {scene_name}, Method: {method}, SSIM: {mean_ssim:.7f}, PSNR: {mean_psnr:.7f}\n")

                    full_dict[scene_dir][method] = {
                        "SSIM": mean_ssim,
                        "PSNR": mean_psnr
                    }
                    per_view_dict[scene_dir][method] = {
                        "SSIM": {name: ssim for ssim, name in zip(ssims, image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(psnrs, image_names)}
                    }

            except Exception as e:
                print(f"Can not evaluate {scene_name}: {e}")
if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
