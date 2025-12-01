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
from numpy.core.fromnumeric import ptp
import torch
from gaussian_renderer import render, render_adaptive
from utils.loss_utils import l2_loss, loss_cls_3d
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, mask_inverse
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import numpy as np
from PIL import Image
    

def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from, out_dir):
    dataset.need_features = False
    dataset.need_masks = False

    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, target='tamper', mode='train')

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    optimization_times = opt.optimization_times

    cams = (scene.getTrainCameras()+scene.getTestCameras())* optimization_times
    coarse_iteration = len(cams) // optimization_times

    mask3d = []
    gt_list = []
    for iteration, view in enumerate(cams):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Render
        if iteration == debug_from:
            pipe.debug = True
        

        if len(gt_list) < coarse_iteration:
            if iteration == 0:
                progress_bar = tqdm(range(len(cams)), desc="tampering localization progress")
            masks = Image.open(out_dir+view.image_name+'.png').convert('L')
            masks = np.array(masks)/255

            gt_mask = torch.from_numpy(masks).float().cuda()
            gt_list.append(gt_mask.unsqueeze(0))
            point_mask = mask_inverse(gaussians._xyz, view, gt_mask)
            mask3d.append(point_mask.unsqueeze(-1))
        else:
            gt_mask = gt_list[iteration % (len(cams) // optimization_times)]

        if iteration == coarse_iteration:
            mask3d = torch.cat(mask3d, dim=1)
            vote_labels,_ = torch.mode(mask3d, dim=1)
            matches = torch.eq(mask3d, vote_labels.unsqueeze(1))
            ratios_mask = torch.sum(matches, dim=1)
            ratios_mask_t = ratios_mask>0
            labels_mask = (vote_labels == 1) & ratios_mask_t
            indices_mask = torch.where(labels_mask)[0].detach().cpu()
            point_mask_grad = torch.full((gaussians._mask.shape[0],), 0).to("cuda")
            ratios_mask = ratios_mask.long()
            point_mask_grad[indices_mask] = ratios_mask[indices_mask]
            gaussians._mask.data += point_mask_grad.unsqueeze(1)

        if iteration >= coarse_iteration:
            render_pkg = render_adaptive(view, gaussians, pipe, background, iteration=iteration, coarse_iteration=coarse_iteration)
            rendered_mask, rendered_depth, viewspace_point_tensor, radii = render_pkg["mask"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["radii"]
            rendered_depth = (rendered_depth - rendered_depth.min()) / (rendered_depth.max() - rendered_depth.min())        
            
            # Loss
            if iteration >= 1.5*coarse_iteration:
                feature = torch.cat((gaussians.get_xyz, gaussians._features_dc.squeeze(1),gaussians.get_opacity, gaussians.get_scaling, gaussians.get_rotation), dim=1)
                loss_3d = loss_cls_3d(feature.detach(), gaussians._mask, gaussians.get_xyz)
                loss = - (gt_mask * rendered_mask).sum() + 10*((1-gt_mask) * rendered_mask).sum() + loss_3d
            else:
                loss = - (gt_mask * rendered_mask).sum() + 10*((1-gt_mask) * rendered_mask).sum()
            loss.backward()

            gaussians._xyz.grad = None
            gaussians._features_dc.grad = None
            gaussians._features_rest.grad = None
            gaussians._scaling.grad = None
            gaussians._rotation.grad = None
            gaussians._opacity.grad = None

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            iter_end.record()
            
        progress_bar.update(1)
    
    progress_bar.close()

    scene.save(iteration, target='tamper')

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--target", type=str, default = 'tamper')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_prompts", default=3, type=int)
    parser.add_argument("--out_dir", type=str, default = '/data/SAFIRE/bicycle/teddybear/')

    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.save_iterations, args.checkpoint_iterations, args.debug_from, args.input_dir, args.out_dir)

    # All done
    print("\nComplete.")