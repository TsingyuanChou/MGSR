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
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from tqdm import tqdm
from collections import deque
from random import randint
from geo_utils.loss_utils import l1_loss, ssim
import geo_gaussian_renderer
import geo_scene
import refgs.ref_scene
from geo_utils.general_utils import safe_state
from geo_utils.image_utils import psnr
import geo_arguments
import refgs.arguments
import refgs.gaussian_renderer
from argparse import ArgumentParser, Namespace
from refgs.utils.loss_utils import l1_loss, ssim, EdgePreservingSmoothnessLoss, SmoothnessLoss, l2_loss, contrastive_loss
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_tv_loss(gt_image: torch.Tensor, prediction: torch.Tensor, pad: int = 1, step: int = 1) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)) 
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True))  
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()
    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(-(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True))  
            rgb_grad_w = torch.exp(-(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True))  
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2) 
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()
    return tv_loss

def training_geo_2d(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,"geo")
    gaussians = geo_scene.GaussianModel(dataset.sh_degree)
    scene = geo_scene.Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print("Loading Checkpoint Iter ",first_iter)
    bg_color = [1, 1, 1] if dataset.geo_white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    loss_window = deque(maxlen = opt.loss_window_size)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = geo_gaussian_renderer.render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mask = viewpoint_cam.mask.cuda()
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        SSIM= 1.0 - ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * SSIM
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_normal if iteration > 3000 else 0.0
        surf_depth = render_pkg['surf_depth']
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        depth_tv_loss = get_tv_loss(gt_image, surf_depth, pad=1, step=1) 
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_error_masked = normal_error * (mask[0].unsqueeze(0))
        normal_edge_loss = torch.tensor(0)
        normal_tv_loss = get_tv_loss(gt_image, rend_normal, pad=1, step=1)
        normal_loss = (lambda_normal * normal_error_masked.mean() + lambda_normal *normal_tv_loss)  + normal_edge_loss
        total_loss = loss + depth_tv_loss + normal_loss
        total_loss.backward()
        iter_end.record()

        if iteration > opt.early_stop_until_iter:
            loss_window.append(total_loss.item())
            if len(loss_window) == opt.loss_window_size:
                changed = max(loss_window) - min(loss_window)
                if changed < opt.threshold:
                    print("Geo stopped at {} iters.".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    break               

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * depth_tv_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report_geo(tb_writer, iteration, Ll1, SSIM, depth_tv_loss,normal_loss, total_loss,iter_start.elapsed_time(iter_end), testing_iterations, scene, geo_gaussian_renderer.render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if iteration < opt.geo_densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.geo_densify_from_iter and iteration % opt.geo_densification_interval == 0:
                    size_threshold = 20 if iteration > opt.geo_opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold) 
                if iteration % opt.geo_opacity_reset_interval == 0 or (dataset.geo_white_background and iteration == opt.geo_densify_from_iter):
                    gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
def training_ref_3d(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,"ref")
    gaussians = refgs.ref_scene.GaussianModel(dataset.sh_degree)
    scene = refgs.ref_scene.Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_window = deque(maxlen = opt.loss_window_size)

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = refgs.gaussian_renderer.render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        ref_map = render_pkg["trans_weights"]
        ref_color = render_pkg["ref_color"]
        trans_color = render_pkg["trans_color"]
        image_python = ref_map * ref_color + trans_color

        if dataset.white_background:
            gt_image = viewpoint_cam.white_bg_image.cuda()
        else:
            gt_image = viewpoint_cam.black_bg_image.cuda()

        depth = render_pkg['out_depth']
        depth_tv_loss = 0.5 * get_tv_loss(gt_image, depth, pad=1, step=1)
        Ll1 = l1_loss(image_python, gt_image)
        SSIM = torch.tensor(1.0) - ssim(image_python, gt_image)
        tran_tv_loss =  get_tv_loss(gt_image, trans_color, pad=1, step=1) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * SSIM + 0.1 * depth_tv_loss + 0.1 * tran_tv_loss

        init_loss = torch.tensor(0)
        depth_smooth_loss = torch.tensor(0)
        ref_map_smooth_loss = torch.tensor(0) 
        loss.backward()
        iter_end.record()
        if iteration > opt.early_stop_until_iter:
            loss_window.append(loss.item())
            if len(loss_window) == opt.loss_window_size:
                changed = max(loss_window) - min(loss_window)
                if changed < opt.threshold:
                    print("Ref stopped at {} iters.".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    break
       
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report_ref(tb_writer, iteration, Ll1,SSIM, init_loss, loss, l1_loss, ref_map_smooth_loss, depth_smooth_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, refgs.gaussian_renderer.render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            if iteration < opt.ref_densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.ref_densify_from_iter and iteration % opt.ref_densification_interval == 0:
                    size_threshold = 20 if iteration > opt.ref_opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.ref_opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.ref_densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
def training_geo_ref(dataset, opt, geo_pipe, ref_pipe, testing_iterations, saving_iterations, checkpoint_iterations, geo_checkpoint,ref_checkpoint):
    first_iter = 0
    tb_writer_geo_ref = prepare_output_and_logger(dataset,"total")
    geo_gaussians = geo_scene.GaussianModel(dataset.sh_degree)
    ge_scene = geo_scene.Scene(dataset, geo_gaussians)
    geo_gaussians.training_setup(opt)

    ref_gaussians = refgs.ref_scene.GaussianModel(dataset.sh_degree)
    re_scene = refgs.ref_scene.Scene(dataset, ref_gaussians)
    ref_gaussians.training_setup(opt)
    if geo_checkpoint:
        (geo_model_params, geo_first_iter) = torch.load(geo_checkpoint)
        geo_gaussians.restore(geo_model_params, opt)
        print("Loading Geo_ckpt")
    if ref_checkpoint:
        (ref_model_params, ref_first_iter) = torch.load(ref_checkpoint)
        ref_gaussians.restore(ref_model_params, opt)
        print("Loading Ref_ckpt")

    ref_bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    geo_bg_color = [1, 1, 1] if dataset.geo_white_background else [0, 0, 0]
    ref_background = torch.tensor(ref_bg_color, dtype=torch.float32, device="cuda")
    geo_background = torch.tensor(geo_bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack_ref = None
    viewpoint_stack_n = None
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0

    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        geo_gaussians.update_learning_rate(iteration)
        ref_gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            geo_gaussians.oneupSHdegree()
            ref_gaussians.oneupSHdegree()

        if not viewpoint_stack_ref:
            viewpoint_stack_ref = re_scene.getTrainCameras().copy()      
        i = randint(0, len(viewpoint_stack_ref)-1)
        viewpoint_cam_ref = viewpoint_stack_ref.pop(i)
        
        render_pkg_geo = geo_gaussian_renderer.render(viewpoint_cam_ref, geo_gaussians, geo_pipe, geo_background)
        image_n, viewspace_point_tensor_n, visibility_filter_n, radii_n = render_pkg_geo["render"], render_pkg_geo["viewspace_points"], render_pkg_geo["visibility_filter"], render_pkg_geo["radii"]
        
        render_pkg_ref = refgs.gaussian_renderer.render(viewpoint_cam_ref, ref_gaussians, ref_pipe, ref_background)
        image_ref, viewspace_point_tensor_ref, visibility_filter_ref, radii_ref ,ref_map,ref_color,trans_color,out_depth_ref = \
            render_pkg_ref["render"], render_pkg_ref["viewspace_points"], render_pkg_ref["visibility_filter"], render_pkg_ref["radii"],\
                 render_pkg_ref["trans_weights"],render_pkg_ref["ref_color"],render_pkg_ref["trans_color"] , render_pkg_ref['out_depth']
       
        mask = viewpoint_cam_ref.mask.cuda()
        black_mask = (mask==0).all(dim=0,keepdim=True)
        if dataset.white_background:
            gt_image_ref = viewpoint_cam_ref.white_bg_image.cuda()
        else:
            gt_image_ref = viewpoint_cam_ref.black_bg_image.cuda()
        
        gt_image_geo = viewpoint_cam_ref.white_bg_image.cuda()
        
        refgs_image = ref_map * ref_color + trans_color
        Ll1_ref = l1_loss(refgs_image, gt_image_ref)
        SSIM_ref = 1.0-ssim(refgs_image, gt_image_ref)
        loss_ref = (1.0 - opt.lambda_dssim) * Ll1_ref + opt.lambda_dssim * SSIM_ref
    
        trans_color_nograd = trans_color.detach()
        trans_color_nograd = torch.where(black_mask,torch.full_like(trans_color_nograd,1.0),trans_color_nograd)
        Ll1_n = opt.w2*l1_loss(image_n, trans_color_nograd) + opt.w1*l1_loss(image_n, gt_image_geo)
        SSIM_n = opt.w2*(1.0-ssim(image_n, trans_color_nograd))+ opt.w1*(1.0-ssim(image_n, gt_image_geo))
        loss_n = (1.0 - opt.lambda_dssim) * Ll1_n + opt.lambda_dssim * SSIM_n

        lambda_normal = opt.lambda_normal
        lambda_dist = opt.lambda_dist
        rend_normal  = render_pkg_geo['rend_normal']
        surf_normal = render_pkg_geo['surf_normal']
        surf_depth = render_pkg_geo['surf_depth']
        depth_tv_loss = lambda_normal * get_tv_loss(trans_color_nograd, surf_depth, pad=1, step=1)
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        
        normal_error_masked = normal_error * (mask[0].unsqueeze(0))
        normal_edge_loss=torch.tensor(0)
        normal_tv_loss = get_tv_loss(trans_color_nograd, rend_normal, pad=1, step=1)      
        normal_loss = (lambda_normal * normal_error_masked.mean() + lambda_normal *normal_tv_loss) + normal_edge_loss
        surf_depth_fixed = surf_depth.detach()
        out_depth_ref = out_depth_ref/out_depth_ref.max()
        surf_depth_fixed = surf_depth_fixed/surf_depth_fixed.max()
        depth_loss =  l2_loss(out_depth_ref, surf_depth_fixed)
        loss_n = loss_n +normal_loss+depth_tv_loss
        total_loss = 0.5 * loss_ref + 0.01 * depth_loss + 0.5 * loss_n 

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss_n.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    #"distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(geo_gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                ge_scene.save(iteration)
                re_scene.save(iteration)
                
            
            #if tb_writer_ref is not None:
                #tb_writer_ref.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                #tb_writer_ref.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report_geo_ref(tb_writer_geo_ref, iteration, loss_ref,loss_n,depth_loss,normal_loss,depth_tv_loss,total_loss, iter_start.elapsed_time(iter_end))
            #if (iteration in saving_iterations):
            #    print("\n[ITER {}] Saving Gaussians".format(iteration))
            #    ref_scene.save(iteration)
                

            # Densification
            if iteration < opt.geo_densify_until_iter:
                geo_gaussians.max_radii2D[visibility_filter_n] = torch.max(geo_gaussians.max_radii2D[visibility_filter_n], radii_n[visibility_filter_n])
                geo_gaussians.add_densification_stats(viewspace_point_tensor_n, visibility_filter_n)

                if iteration > opt.geo_densify_from_iter and iteration % opt.geo_densification_interval == 0:
                    size_threshold = 20 if iteration > opt.geo_opacity_reset_interval else None
                    geo_gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, ge_scene.cameras_extent, size_threshold)
                
                if iteration % opt.geo_opacity_reset_interval == 0 or (dataset.geo_white_background and iteration == opt.geo_densify_from_iter):
                    geo_gaussians.reset_opacity()
            if iteration < opt.ref_densify_until_iter:
                # Keep track of max radii in image-space for pruning
                ref_gaussians.max_radii2D[visibility_filter_ref] = torch.max(ref_gaussians.max_radii2D[visibility_filter_ref], radii_ref[visibility_filter_ref])
                ref_gaussians.add_densification_stats(viewspace_point_tensor_ref, visibility_filter_ref)


                if iteration > opt.ref_densify_from_iter and iteration % opt.ref_densification_interval == 0:
                    size_threshold = 20 if iteration > opt.ref_opacity_reset_interval else None
                    ref_gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, re_scene.cameras_extent, size_threshold)
                
                if iteration % opt.ref_opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.ref_densify_from_iter):
                    ref_gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                geo_gaussians.optimizer.step()
                geo_gaussians.optimizer.zero_grad(set_to_none = True)
                ref_gaussians.optimizer.step()
                ref_gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((geo_gaussians.capture(), iteration), ge_scene.model_path + "/chkpnt" + str(iteration) + ".pth")                

def prepare_output_and_logger(args,module):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/{}".format(module), unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report_geo_ref(tb_writer, iteration, loss_ref,loss_n,depth_loss,normal_loss,dist_loss,total_loss, elapsed):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/loss_ref', loss_ref.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_n', loss_n.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/dist_loss', dist_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

@torch.no_grad()
def training_report_geo(tb_writer, iteration, Ll1, SSIM, dist_loss,normal_loss, total_loss, elapsed, testing_iterations, scene : geo_scene.Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/Ll1', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/SSIM', SSIM.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/dist_loss', dist_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    '''
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from geo_utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()
    '''
        
def training_report_ref(tb_writer, iteration, Ll1,SSIM, init_loss, loss, l1_loss, ref_map_smooth_loss, depth_smooth_loss, elapsed, testing_iterations, scene : refgs.ref_scene.Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/SSIM', SSIM.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/init_loss', init_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ref_map_smooth_loss', ref_map_smooth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_smooth_loss', depth_smooth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    '''
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in [23, 24, 30, 37, 45]]})
        # typical_test_cameras = ['DSCF4707', 'DSCF4731', 'DSCF4819', 'DSCF4827', 'DSCF4851']
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_package = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_package["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.black_bg_image.to("cuda"), 0.0, 1.0)
                    depth = render_package["out_depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())

                    ref_map = torch.clamp(render_package["trans_weights"], 0.0, 1.0)
                    ref_color = torch.clamp(render_package["ref_color"], 0.0, 1.0)
                    trans_color = torch.clamp(render_package["trans_color"], 0.0, 1.0)
                    comp_ref_color = ref_map * ref_color
                    image_python = trans_color + comp_ref_color
                    # if tb_writer and ((config['name'] == 'test' and viewpoint.image_name in typical_test_cameras) or 
                    #                   (config['name'] == 'train' and idx < 5)):
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)                        # tb_writer.add_images(config['name'] + "_view_{}/render_python".format(viewpoint.image_name), image_python[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ref_map".format(viewpoint.image_name), ref_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ref_color".format(viewpoint.image_name), ref_color[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/trans_color".format(viewpoint.image_name), trans_color[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/comp_ref_color".format(viewpoint.image_name), comp_ref_color[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    '''

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp_geo = geo_arguments.ModelParams(parser)
    op_geo = geo_arguments.OptimizationParams(parser)
    pp_geo = geo_arguments.PipelineParams(parser)
    pp_ref = refgs.arguments.PipelineParams(parser)
    op_geo.iterations = 20_000
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)
    path = args.model_path
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    #Geo training
    args.iterations = 20_000
    args.checkpoint_iterations=[20_000]
    args.model_path = path +os.sep+"geo"
    training_geo_2d(lp_geo.extract(args), op_geo.extract(args), pp_geo.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)
    
    #Ref training
    args.iterations = 20_000
    args.checkpoint_iterations=[20_000]
    args.model_path = path +os.sep+"ref"
    training_ref_3d(lp_geo.extract(args), op_geo.extract(args), pp_ref.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)
    
    saved_ref_checkpoint = glob.glob(path +os.sep+"ref"+os.sep+"*.pth",recursive=True)
    saved_geo_checkpoint = glob.glob(path +os.sep+"geo"+os.sep+"*.pth",recursive=True)
    args.iterations = 20_000
    args.save_iterations.append(20_000)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations=[]
    args.model_path = path +os.sep+"total"
    geo_checkpoint = saved_geo_checkpoint[0]
    ref_checkpoint = saved_ref_checkpoint[0]
    training_geo_ref(lp_geo.extract(args), op_geo.extract(args),pp_geo.extract(args), pp_ref.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, geo_checkpoint,ref_checkpoint)
    print("\nTraining complete.")