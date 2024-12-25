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
import os
import time
from datetime import datetime
from pathlib import Path

import keyboard
from devices.utils import fps_wait
from devices.constants import BUTTON_MAP_KEY
from devices import CameraGroup, XRobot
import hydra
from omegaconf import DictConfig

from lerobot.devices import build_robot


class Recorder:

    def __init__(self, cfg: DictConfig, robot: XRobot):
        self.save_path = os.path.join(cfg.task.record_dir, datetime.now().strftime("%m_%d"))
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.robot = robot
        self.camera = CameraGroup(cfg.task.camera_names, cfg.task.image_shape[1], cfg.task.image_shape[2])
        self.fps = self.cfg.fps
        self.record_frequency = self.cfg.frequency
        print("Moving FPS", self.fps, "Recording FPS", self.fps / self.record_frequency)
        self.robot.enable_robot()

    def __del__(self):
        self.robot.robot_follow_disable()
        time.sleep(1)
        self.robot.disable_robot()

    def record(self):
        i = 0
        print("enable follow?")
        keyboard.wait(BUTTON_MAP_KEY)

        keyboard.on_press_key("num lock", _change_running_flag)
        self.robot.robot_follow_enable()
        while True:
            self.record_one()
            i += 1
            print('next episode？:', i)

    def _record_episode(self):
        start = time.time()
        camera = self.camera.read(self.cfg.task.camera_names)
        action, _, master_gripper = self.robot.robot_get_master_angle()
        angles, _, gripper = self.robot.robot_get_angle_velocity()

        print("master", action, master_gripper)
        print("follow", angles, gripper)
        episode = {"camera": camera,
                   "right_master": action + [master_gripper],
                   "right_puppet": angles + [gripper]}
        fps_wait(self.fps, start)
        return episode

    def record_one(self):
        print('start record now?')
        keyboard.wait(BUTTON_MAP_KEY)
        episodes = []

        for i in range(3):
            images = self.camera.read_sync()

        start_tm = time.time()
        i = 0
        while RECORD_FLAG:
            st = time.time()
            episode = self._record_episode()
            fps_wait(self.fps, st)
            if i % self.record_frequency == 0:
                episodes.append(episode)
            i += 1

        duration = time.time() - start_tm
        f = os.path.join(self.save_path, f"{datetime.now().strftime('%m_%d_%H_%M_%S')}.pkl")
        pickle.dump({"data": episodes, "task": self.cfg.task.name, "fps": self.fps / self.record_frequency},
                    open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)} FPS {round(len(episodes) / duration, 2)}')


RECORD_FLAG = False


def _change_running_flag(event):
    global RECORD_FLAG
    RECORD_FLAG = not RECORD_FLAG
    print(f"change running flag to {RECORD_FLAG}")


@hydra.main(version_base="1.2", config_name="coffee", config_path="configs/coffee")
def run(cfg: DictConfig):
    print(cfg)
    robot = XRobot()
    r = Recorder(cfg, robot)
    r.record()


if __name__ == '__main__':
    # task_name = sys.argv[1]
    # arm_left, arm_right = build_two_arm()
    # r = Recorder(arm_left, arm_right)
    # r.record()
    run()
