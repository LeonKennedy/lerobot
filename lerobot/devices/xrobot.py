import time
from typing import List

import can
from loguru import logger
import keyboard


class Joint:

    def __init__(self, joint_id, pmax: float = 12.5, vmax: float = 10, tmax: float = 28, rad_min: float = -10,
                 rad_max: float = 10):
        self.id = joint_id

        self.angle = 0
        self.velocity = 0
        self.enabled = False

        self.pmax = pmax
        self.vmax = vmax
        self.tmax = tmax
        self.rad_min = rad_min
        self.rad_max = rad_max


class Robot:
    COMMAND_ENABLE_OR_DISABLE = 0x101
    COMMAND_MOVE_JOINT_TARGET_ANGLE = 0x202
    COMMAND_GET_ROBOT_ANGLE_VEL = 0x112
    COMMAND_FOLLOW_MODE = 0x104
    COMMAND_FOLLOW_ANGLE = 0x105
    COMMAND_GET_MASTER_ANGLE = 0x333
    COMMAND_SET_JOINT_ZERO = 0x334

    def __init__(self):
        self.bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", fd=True, f_clock=80000000,
                                     nom_brp=1,  # Nominal Bit Rate Prescaler
                                     nom_tseg1=59,  # Nominal Time Segment 1 (quanta before sample point)
                                     nom_tseg2=20,  # Nominal Time Segment 2 (quanta after sample point)
                                     nom_sjw=20,  # Synchronization Jump Width for nominal bit rate
                                     data_brp=1,  # Data Bit Rate Prescaler
                                     data_tseg1=13,  # Data Time Segment 1 for data phase
                                     data_tseg2=1,  # Data Time Segment 2 for data phase
                                     data_sjw=2)
        self.joint_0 = Joint(0)
        self.joint_1 = Joint(1)
        self.joint_2 = Joint(2)
        self.joint_3 = Joint(3)
        self.joint_4 = Joint(4, vmax=30, tmax=10)
        self.joint_5 = Joint(5, vmax=30, tmax=10)
        self.joints = {
            0: self.joint_0,
            1: self.joint_1,
            2: self.joint_2,
            3: self.joint_3,
            4: self.joint_4,
            5: self.joint_5
        }

        self._check()
        self._is_follow = False

    def __del__(self):
        self.bus.shutdown()

    @property
    def is_follow(self) -> bool:
        return self._is_follow

    @staticmethod
    def __uint_to_float(x_int, x_min, x_max, bits):
        span = x_max - x_min
        offset = x_min
        return (x_int * span / ((1 << bits) - 1)) + offset

    @staticmethod
    def __float_to_uint(x, x_min, x_max, bits):
        span = x_max - x_min
        offset = x_min
        uint_value = int((x - offset) * ((1 << bits) - 1) / span)
        # print(x, uint_value)
        # 使用格式化字符串将整数转换为指定位数的十六进制字符串
        high_byte = (uint_value >> 8) & 0xFF
        low_byte = uint_value & 0xFF
        # hex_value = format(uint_value, f'0{bits // 4}X')
        # print(hex_value)
        return high_byte, low_byte

    def __send_msg(self, arbitration_id: int, data: List[int]):
        # print("send_msg:", data)
        msg = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False, is_fd=True)
        self.bus.send(msg)

    def __recv_msg(self):
        return self.bus.recv(5)

    def enable_robot(self):
        self.__send_msg(Robot.COMMAND_ENABLE_OR_DISABLE, [0x01] * 6)
        logger.debug("robot enable")

    def disable_robot(self):
        self.__send_msg(Robot.COMMAND_ENABLE_OR_DISABLE, [0x00] * 6)
        logger.debug("robot disable")

    def robot_follow_enable(self):
        self.__send_msg(Robot.COMMAND_FOLLOW_MODE, [0x01])
        self._is_follow = True
        logger.debug("robot follow start")

    def robot_follow_disable(self):
        self.__send_msg(Robot.COMMAND_FOLLOW_MODE, [0x00])
        self._is_follow = False
        logger.debug("robot follow stop")

    def move_to_zero(self):
        # self.__send_msg()
        pass

    def joint_move_angle(self, joint_id, target_angle, velocity: float = 2):
        data = [joint_id]
        data.extend(self.__float_to_uint(target_angle, -self.joints[joint_id].pmax, self.joints[joint_id].pmax, 16))
        data.extend(self.__float_to_uint(velocity, -self.joints[joint_id].vmax, self.joints[joint_id].vmax, 16))
        # print(data)
        self.__send_msg(Robot.COMMAND_MOVE_JOINT_TARGET_ANGLE, data)

    def follow_angle(self, angle):
        data = []
        for joint_id in range(0, 6):
            data.extend(
                self.__float_to_uint(angle[joint_id], -self.joints[joint_id].pmax, self.joints[joint_id].pmax, 16))
        # print(data)
        self.__send_msg(Robot.COMMAND_FOLLOW_ANGLE, data)

    def robot_get_angle_velocity(self):
        self.__send_msg(Robot.COMMAND_GET_ROBOT_ANGLE_VEL, [])
        response = self.__recv_msg()
        print(response)
        angles = []
        vels = []
        if response.arbitration_id == 0x112:
            data = response.data.hex()
            # print(data, len(data))
            for i in range(6):
                angle = (int(data[i * 4:i * 4 + 4], 16) - 2 ** 15) / 2 ** 16 * self.joints[i].pmax * 2
                vel = (int(data[i * 4 + 24:i * 4 + 4 + 24], 16) - 2 ** 15) / 2 ** 16 * self.joints[i].vmax * 2
                # print(i, angle, vel)
                angles.append(angle)
                vels.append(vel)
            # print(data[48:48+4], int(data[48:48+4], 16))
            gripper = int(data[48:48 + 4], 16)
        return angles, vels, gripper

    def robot_get_master_angle(self):
        self.__send_msg(Robot.COMMAND_GET_MASTER_ANGLE, [])
        response = self.__recv_msg()
        print(response)
        angles = []
        vels = []
        gripper = None
        if response.arbitration_id == Robot.COMMAND_GET_MASTER_ANGLE:
            data = response.data.hex()
            # print(data, len(data))
            for i in range(6):
                angle = (int(data[i * 4:i * 4 + 4], 16) - 2 ** 15) / 2 ** 16 * self.joints[i].pmax * 2
                vel = (int(data[i * 4 + 24:i * 4 + 4 + 24], 16) - 2 ** 15) / 2 ** 16 * self.joints[i].vmax * 2
                # print(i, angle, vel)
                angles.append(angle)
                vels.append(vel)
            print(data[48:48 + 4], int(data[48:48 + 4], 16))
            gripper = int(data[48:48 + 4], 16)
        return angles, vels, gripper

    def set_joint_zero(self, joint_id):
        data = [joint_id]
        self.__send_msg(Robot.COMMAND_SET_JOINT_ZERO, data)

    # def robot_move(self, target_angle_list: List[float], velocity_list: List[float]):
    # data = [self.__float_to_uint(target_angle)]

    def _check(self):
        angles, vels, gripper = self.robot_get_angle_velocity()
        logger.debug(f"checking, start at {angles}")
        all_small_diff = True
        for i in range(len(angles) - 1):
            if abs(angles[i] - angles[i + 1]) >= 0.01:
                all_small_diff = False
                break
        if all_small_diff:
            logger.error("所有角度差值都小于0.01，退出程序")
            exit()


class XRobot(Robot):

    def disable(self):
        self.disable_robot()

    def enable(self):
        self.enable_robot()


if __name__ == '__main__':
    robot = XRobot()


    def _change_follow(event):
        if robot.is_follow:
            robot.robot_follow_disable()
        else:
            robot.robot_follow_enable()


    keyboard.on_press_key("num lock", _change_follow)
