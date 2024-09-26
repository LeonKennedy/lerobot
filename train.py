#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: train.py
@time: 2024/9/12 11:40
@desc:
"""
import os
from pathlib import Path
from loguru import logger
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from preprocess_data.load_dataset import build_dataset, build_act_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)

def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "act":
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_params_dicts, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        lr_scheduler = None
    elif cfg.policy.name == "diffusion":
        optimizer = torch.optim.Adam(
            policy.diffusion.parameters(),
            cfg.training.lr,
            cfg.training.adam_betas,
            cfg.training.adam_eps,
            cfg.training.adam_weight_decay,
        )
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    elif policy.name == "tdmpc":
        optimizer = torch.optim.Adam(policy.parameters(), cfg.training.lr)
        lr_scheduler = None
    elif cfg.policy.name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTOptimizer, VQBeTScheduler

        optimizer = VQBeTOptimizer(policy, cfg)
        lr_scheduler = VQBeTScheduler(optimizer, cfg)
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler

@hydra.main(version_base='1.2', config_name="coffee", config_path="lerobot/configs/coffee")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    output_directory = Path(cfg.dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    data_path = cfg.data_path
    if cfg.policy.name == "act":
        dataset, stats = build_act_dataset(data_path, cfg.policy.n_action_steps)
    elif cfg.policy.name == "diffusion":
        dataset, stats = build_dataset(data_path,
                                       cfg.policy.n_obs_steps,
                                       cfg.policy.horizon)
    logger.info(f"loaded dataset: {data_path} {len(dataset)}")
    policy = make_policy(hydra_cfg=cfg,
                         dataset_stats= stats if not cfg.resume else None,
                         pretrained_policy_name_or_path=str(output_directory) if cfg.resume else None)
    policy.train()
    policy.to(device)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    logger.info("created dataloader")

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % cfg.training.log_freq == 0:
                logger.debug(f"step: {step} loss: {loss.item():.3f}")

            if cfg.training.save_checkpoint and step % cfg.training.save_freq == 0:
                policy.save_pretrained(output_directory)
            step += 1
            if step >= cfg.training.offline_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == '__main__':
    run()
