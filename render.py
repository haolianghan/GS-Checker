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

from sys import platform
import torch
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
import numpy as np
from gaussian_renderer import render
import torchvision
import matplotlib.pyplot as plt
import json
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene_name, MASK_THRESHOLD):
    render_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "renders")
    mask_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "mask")

    makedirs(render_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    render_func = render

    with open(f'./test/{scene_name}.json', 'r') as f:
        test_data = json.load(f)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if view.image_name in test_data:
            res = render_func(view, gaussians, pipeline, background)
            rendering = res["render"]
                        
            mask = res["mask"]
            mask = mask.squeeze()
            
            norm = mask / mask.max()
            mask[norm <= MASK_THRESHOLD] = 0.
            mask[norm > MASK_THRESHOLD] = 1.
           
            torchvision.utils.save_image(mask, os.path.join(mask_path, f"{view.image_name}.png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, target = 'tamper', scene_name='bicycle', MASK_THRESHOLD=0.1):
    dataset.need_features = dataset.need_masks = False
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval', target=target)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras()+scene.getTestCameras(), gaussians, pipeline, background, scene_name, MASK_THRESHOLD)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--target', default='tamper')
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--scene_name', default='bicycle', type=str)
    parser.add_argument('--MASK_THRESHOLD', default=0.1, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.target, args.scene_name, args.MASK_THRESHOLD)