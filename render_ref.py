#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import torch
from refgs.ref_scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from refgs.gaussian_renderer import render
import torchvision
from refgs.utils.general_utils import safe_state
from argparse import ArgumentParser
from refgs.arguments import ModelParams, PipelineParams, get_combined_args
from refgs.utils.loss_utils import l1_loss, ssim, EdgePreservingSmoothnessLoss, SmoothnessLoss, l2_loss, contrastive_loss
from refgs.gaussian_renderer import GaussianModel
import numpy as np
import cv2
import math
from geo_utils.image_utils import psnr
from refgs.utils.pose_utils import generate_ellipse_path, generate_spiral_path
from refgs.utils.graphics_utils import getWorld2View2
from geo_utils.point_utils import depth_to_normal
from geo_utils.render_utils import save_img_f32, save_img_u8

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    ref_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ref_map")
    ref_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ref_color")
    trans_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "trans_color")
    comp_ref_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "comp_ref_color")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(ref_map_path, exist_ok=True)
    makedirs(ref_color_path, exist_ok=True)
    makedirs(trans_color_path, exist_ok=True)
    makedirs(comp_ref_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    t_list = []
    l1_test = 0.0
    ssim_test = 0.0
    psnr_test = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t0 = time.time()
        render_package = render(view, gaussians, pipeline, background)
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1 - t0)
        print(f'Render time: \033[1;35m{t1 - t0:.5f}\033[0m')
        rendering = render_package["render"]
        ref_map = render_package["trans_weights"]
        ref_color = render_package["ref_color"]
        trans_color = render_package["trans_color"]
        comp_ref_color = ref_map * ref_color
        depth = render_package["out_depth"]

        #gt = view.black_bg_image[0:3, :, :]
        gt = view.white_bg_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ref_map, os.path.join(ref_map_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ref_color, os.path.join(ref_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(trans_color, os.path.join(trans_color_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(comp_ref_color, os.path.join(comp_ref_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        
        
        l1_test += l1_loss(rendering, gt).mean().double()
        ssim_test += ssim(rendering, gt).mean().double()
        psnr_test += psnr(rendering, gt).mean().double()
    t = np.array(t_list)
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
    print("l1",l1_test.item()/100)
    print("ssim",ssim_test/100)
    print("psnr",psnr_test/100)
    
def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, split, camera_type, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]

    if camera_type == 'ellipse': 
        render_poses = generate_ellipse_path(views)
    elif camera_type == 'spiral':
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'), n_frames=600)
    else: 
        print("Camera type not supported!")
        return

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    final_video = cv2.VideoWriter(os.path.join(render_path, f'{split}_video_{camera_type}.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        # torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.render_video:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                         gaussians, pipeline, background, "test", args.camera_type, args.fps)
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                         gaussians, pipeline, background, "train", args.camera_type, args.fps)
                         
        if args.render_images: 
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

            if not skip_test: 
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=20000, type=int)
    parser.add_argument("--render_images", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--camera_type", default='spiral', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)