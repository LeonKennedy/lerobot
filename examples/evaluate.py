#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: evaluate.py
@time: 2024/9/10 17:38
@desc:
"""
import os.path
import time
from pathlib import Path
import numpy
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.devices import build_right_arm, CameraGroup
from lerobot.devices.utils import fps_wait

right_arm = build_right_arm()
right_arm.move_start_position(master=False)
camera = CameraGroup()

pretrained_policy_path = "../outputs/train/example_pusht_diffusion"
assert os.path.exists(pretrained_policy_path)
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Device set to:", device)
else:
    device = torch.device("cpu")
    print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
    policy.diffusion.num_inference_steps = 10

policy.to(device)
# Reset the policy and environmens to prepare for rollout
policy.reset()
# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state

step = 0
done = False
while not done:
    # Prepare observation for the policy running in Pytorch
    start = time.time()
    puppet_state = right_arm.read_puppet_state()
    img = camera.read_right()
    state = torch.tensor(puppet_state, dtype=torch.float32)
    image = torch.from_numpy(img)
    image = torch.einsum('h w c -> c h w', image)
    image = image.to(torch.float32)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()

    fps_wait(15, start)
    print("OUT:", numpy_action)
    bit_width = 1 / (time.time() - start) / 2
    right_angle, right_grasper = numpy_action[:6], numpy_action[6]
    right_arm.puppet.move_to(right_angle, bit_width)
    right_arm.grasper.set_angle(right_grasper)
    step += 1


# Encode all frames into a mp4 video.

# video_path = output_directory / "rollout.mp4"
# imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
#
# print(f"Video of the evaluation is available in '{video_path}'.")
