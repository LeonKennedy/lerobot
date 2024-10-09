#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: record.py
@time: 2024/5/20 15:52
@desc:
"""
import pickle
import sys
import os
import time
from datetime import datetime
from pathlib import Path

import keyboard
from typing import List, Optional

from devices.utils import fps_wait
from devices.constants import BUTTON_MAP_KEY
from devices import CameraGroup, build_two_arm, Arm, build_right_arm, Robot
import hydra
from omegaconf import DictConfig

from lerobot.devices import build_robot


class Recorder:

    def __init__(self, cfg: DictConfig, robot: Robot):
        self.save_path = os.path.join(cfg.task.record_dir, datetime.now().strftime("%m_%d"))
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.robot = robot
        self.camera = CameraGroup()
        self.fps = self.cfg.fps
        self.bit_width = 1 / self.fps / 2
        self.record_frequency = self.cfg.frequency
        print("Moving FPS", self.fps, "Recording FPS", self.fps / self.record_frequency)

    def clear_uart(self):
        self.robot.clear_uart()

    def set_end_torque_zero(self):
        self.robot.set_end_torque_zero()

    def record(self):
        k = input('[DO FIRST]\n1. two arm move to start position?\n2. master move to puppet?(q)')
        if k == '1':
            self.robot.move_start_position()
        elif k == '2':
            self.robot.move_master_to_puppet()
        else:
            pass

        self.set_end_torque_zero()
        print("move done, set end torque zero..")
        self.clear_uart()
        i = 0
        global RUNNING_FLAG
        keyboard.on_press_key(BUTTON_MAP_KEY, _change_running_flag)
        while True:
            self.record_one()
            i += 1
            RUNNING_FLAG = True
            self.follow()
            print('next episode？:', i)
            self.clear_uart()

    def _record_episode(self, info=True):
        start = time.time()
        episode = self.robot.follow(self.bit_width)

        tm1 = time.time()
        episode["camera"] = self.camera.read(self.cfg.task.camera_names)
        camera_cost = time.time() - tm1

        fps_wait(self.fps, start)
        duration = time.time() - start
        self.bit_width = 1 / duration / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间

        if info:
            print(duration, "bit_width:", self.bit_width, "camera:", round(camera_cost, 4))
            # print("left", episode["left_master"], episode["left_puppet"])
            # print("right", episode["right_master"], episode["right_puppet"])
        return episode

    def record_one(self):
        print('start record now?')
        keyboard.wait(BUTTON_MAP_KEY)
        episodes = []

        for i in range(3):
            images = self.camera.read_sync()

        start_tm = time.time()
        i = 0
        while RUNNING_FLAG:
            episode = self._record_episode()
            if i % self.record_frequency == 0:
                episodes.append(episode)
            i += 1

        duration = time.time() - start_tm
        f = os.path.join(self.save_path, f"{datetime.now().strftime('%m_%d_%H_%M_%S')}.pkl")
        pickle.dump({"data": episodes, "task": self.cfg.task.name, "fps": self.fps / self.record_frequency},
                    open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)} FPS {round(len(episodes) / duration, 2)}')

    def follow(self):
        while RUNNING_FLAG:
            self._record_episode(False)

        self.robot.lock()


RUNNING_FLAG = False


def _change_running_flag(event):
    global RUNNING_FLAG
    RUNNING_FLAG = not RUNNING_FLAG
    print(f"change running flag to {RUNNING_FLAG}")


@hydra.main(version_base="1.2", config_name="coffee", config_path="configs/coffee")
def run(cfg: DictConfig):
    print(cfg)
    robot = build_robot(cfg.task.action_dim)
    r = Recorder(cfg, robot)
    r.record()


if __name__ == '__main__':
    # task_name = sys.argv[1]
    # arm_left, arm_right = build_two_arm()
    # r = Recorder(arm_left, arm_right)
    # r.record()
    run()
