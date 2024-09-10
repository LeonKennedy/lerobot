#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: load_dataset.py
@time: 2024/9/10 11:54
@desc:
"""
import glob
import os
import pickle
from typing import List

import torch
import numpy as np
from loguru import logger

_BASE_PATH = "/mnt/d4t/code/act_custom"


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, data: dict, obs_horizon: int, pred_horizon: int):
        super(EpisodicDataset).__init__()
        self.action, self.state, self.image = self._preprocess(data)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

    def _preprocess(self, data):
        master = []
        puppet = []
        images = []
        for e in data:
            master.append(e['right_master'])
            puppet.append(e['right_puppet'])
            images.append(e['camera']['RIGHT'])

        images = np.stack(images)
        images = np.moveaxis(images, -1, 1)
        return np.stack(master), np.stack(puppet), images

    def __len__(self) -> int:
        # return self.data['qpos'].shape[0] - 20
        return 10

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.action.shape[0] - int(self.pred_horizon * 0.5))
        imgs = self.image[idx: idx + self.obs_horizon]
        state = self.state[idx: idx + self.obs_horizon]
        is_pad = np.zeros(self.pred_horizon)
        if idx + self.obs_horizon + self.pred_horizon > self.action.shape[0]:
            padding_action = np.zeros((self.pred_horizon, self.action.shape[1]), dtype=np.float32)
            actual_len = self.action.shape[0] - idx - self.obs_horizon
            padding_action[:actual_len] = self.action[idx + self.obs_horizon:]
            is_pad[actual_len:] = 1
        else:
            padding_action = self.action[idx + self.obs_horizon: idx + self.obs_horizon + self.pred_horizon]

        out = {
            'observation.image': imgs.astype(np.float32),  # (B,obs,c,h,w)
            'observation.state': state.astype(np.float32),  # (B,obs,dim)
            'action': padding_action.astype(np.float32),  # (B,pre,dim)
            'action_is_pad': is_pad.astype(np.bool_)
        }
        return out

    def add_noise(self, pos):
        return pos + np.random.normal(0, 0.01, pos.shape)


def _get_data_stats(data):
    stats = {
        'min': torch.from_numpy(np.min(data, axis=0)).float(),
        'max': torch.from_numpy(np.max(data, axis=0)).float(),
        'mean': torch.from_numpy(np.mean(data, axis=0)).float(),
        'std': torch.from_numpy(np.std(data, axis=0)).float()
    }
    return stats


def _image_stats(data):
    if 1:
        return {"mean": torch.tensor([96.5238, 97.9138, 94.9040]).reshape(3, 1, 1),
                "std": torch.tensor([55.3451, 58.1282, 54.7187]).reshape(3, 1, 1)}
    else:
        image = np.concatenate([d.image for d in data])
        mean = []
        std = []
        for i in range(3):
            m = image[:, i, :, :].mean()
            s = image[:, i, :, :].std()
            mean.append(m)
            std.append(s)
        return {"mean": torch.tensor(mean).reshape(3, 1, 1), "std": torch.tensor(std).reshape(3, 1, 1)}


def get_stats(data: List[EpisodicDataset]):
    out = {}
    action = np.concatenate([d.action for d in data])
    out['action'] = _get_data_stats(action)
    logger.debug("finish action stats")
    state = np.concatenate([d.state for d in data])
    out['observation.state'] = _get_data_stats(state)
    logger.debug("finish state stats")
    out['observation.image'] = _image_stats(data)
    return out


def build_dataset(name: str, obs_horizon: int = 2, pred_horizon: int = 16):
    path = os.path.join(_BASE_PATH, "output", name)
    out = []
    for i in glob.glob(os.path.join(path, "*.pkl")):
        data = pickle.load(open(i, 'rb'))
        if "right" in data["task"]:
            ed = EpisodicDataset(data["data"], obs_horizon, pred_horizon)
            out.append(ed)
    logger.debug("finish concat dataset")
    stats = get_stats(out)
    print(stats)
    return torch.utils.data.ConcatDataset(out), stats


if __name__ == '__main__':
    ds = build_dataset("cube")
    print(ds[0])
