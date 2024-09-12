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
from devices.constants import FPS, BUTTON_MAP_KEY
from devices import CameraGroup, build_two_arm, Arm, build_right_arm
import hydra
from omegaconf import DictConfig


class Recorder:

    def __init__(self, cfg: DictConfig, arm_left: Optional[Arm] = None, arm_right: Optional[Arm] = None):
        self.save_path = os.path.join(cfg.task.dir, datetime.now().strftime("%m_%d"))
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.arm_left = arm_left
        self.arm_right = arm_right
        self.camera = CameraGroup()
        self.bit_width = 1 / FPS / 2
        self.record_frequency = self.cfg.frequency
        print("Moving FPS", FPS, "Recording FPS", FPS / self.record_frequency)

    def clear_uart(self):
        if self.arm_left:
            self.arm_left.clear_uart()
        if self.arm_right:
            self.arm_right.clear_uart()

    def move_start(self):
        if self.arm_left:
            self.arm_left.move_start_position()
        if self.arm_right:
            self.arm_right.move_start_position()

    def master_to_puppet(self):
        if self.arm_left:
            lm, lp = self.arm_left.get_all_angle()
            self.arm_left.master.move_to1(lp)
        if self.arm_right:
            rm, rp = self.arm_right.get_all_angle()
            self.arm_right.master.move_to1(rp)

    def set_end_torque_zero(self):
        if self.arm_left:
            self.arm_left.master.set_end_torque_zero()
        if self.arm_right:
            self.arm_right.master.set_end_torque_zero()

    def record(self):
        k = input('[DO FIRST]\n1. two arm move to start position?\n2. master move to puppet?(q)')
        if k == '1':
            self.move_start()
        elif k == '2':
            self.master_to_puppet()
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
        episode = {}
        if self.arm_left:
            left_master_angles, left_trigger_angle, left_puppet_angles, left_grasper_angle = self.arm_left.follow(
                self.bit_width)
            episode["left_master"] = left_master_angles + [left_trigger_angle]
            episode["left_puppet"] = left_puppet_angles + [left_grasper_angle]
        if self.arm_right:
            right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = self.arm_right.follow(
                self.bit_width)
            episode["right_master"] = right_master_angles + [right_trigger_angle]
            episode["right_puppet"] = right_puppet_angles + [right_grasper_angle]

        tm1 = time.time()
        episode["camera"] = self.camera.read(self.cfg.camera_names)
        camera_cost = time.time() - tm1

        fps_wait(FPS, start)
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
        pickle.dump({"data": episodes, "task": self.cfg.task.name}, open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)} FPS {round(len(episodes) / duration, 2)}')

    def follow(self):
        while RUNNING_FLAG:
            self._record_episode(False)

        if self.arm_left:
            self.arm_left.lock()
        if self.arm_right:
            self.arm_right.lock()


RUNNING_FLAG = False


def _change_running_flag(event):
    global RUNNING_FLAG
    RUNNING_FLAG = not RUNNING_FLAG
    print(f"change running flag to {RUNNING_FLAG}")


@hydra.main(version_base="1.2", config_name="record", config_path="configs/coffee")
def run(cfg: DictConfig):
    print(cfg)
    arm_right = build_right_arm()
    r = Recorder(cfg, arm_right=arm_right)
    r.record()


if __name__ == '__main__':
    # task_name = sys.argv[1]
    # arm_left, arm_right = build_two_arm()
    # r = Recorder(arm_left, arm_right)
    # r.record()
    run()
