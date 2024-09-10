#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: daran.py
@time: 2024/8/27 16:44
@desc:
"""

import numpy as np


class DaranMotorsBus:
    """
    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "xl330-m288"

    motors_bus = DynamixelMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
            self,
            port: str,
            motors: dict[str, tuple[int, str]],
            extra_model_control_table: dict[str, list[tuple]] | None = None,
            extra_model_resolution: dict[str, int] | None = None,
    ):
        self.port = port
        self.motors = motors

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        if extra_model_control_table:
            self.model_ctrl_table.update(extra_model_control_table)

        self.model_resolution = deepcopy(MODEL_RESOLUTION)
        if extra_model_resolution:
            self.model_resolution.update(extra_model_resolution)

        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"DynamixelMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/common/robot_devices/motors/dynamixel.py` to make sure you are using the correct port.\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

        # Set expected baudrate for the bus
        self.set_bus_baudrate(BAUDRATE)

        if not self.are_motors_configured():
            input(
                "\n/!\\ A configuration issue has been detected with your motors: \n"
                "If it's the first time that you use these motors, press enter to configure your motors... but before "
                "verify that all the cables are connected the proper way. If you find an issue, before making a modification, "
                "kill the python process, unplug the power cord to not damage the motors, rewire correctly, then plug the power "
                "again and relaunch the script.\n"
            )
            print()
            self.configure_motors()

    def reconnect(self):
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")
        self.is_connected = True

    def are_motors_configured(self):
        # Only check the motor indices and not baudrate, since if the motor baudrates are incorrect,
        # a ConnectionError will be raised anyway.
        try:
            return (self.motor_indices == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

    def configure_motors(self):
        # TODO(rcadene): This script assumes motors follow the X_SERIES baudrates
        # TODO(rcadene): Refactor this function with intermediate high-level functions

        print("Scanning all baudrates and motor indices")
        all_baudrates = set(X_SERIES_BAUDRATE_TABLE.values())
        ids_per_baudrate = {}
        for baudrate in all_baudrates:
            self.set_bus_baudrate(baudrate)
            present_ids = self.find_motor_indices()
            if len(present_ids) > 0:
                ids_per_baudrate[baudrate] = present_ids
        print(f"Motor indices detected: {ids_per_baudrate}")
        print()

        possible_baudrates = list(ids_per_baudrate.keys())
        possible_ids = list({idx for sublist in ids_per_baudrate.values() for idx in sublist})
        untaken_ids = list(set(range(MAX_ID_RANGE)) - set(possible_ids) - set(self.motor_indices))

        # Connect successively one motor to the chain and write a unique random index for each
        for i in range(len(self.motors)):
            self.disconnect()
            input(
                "1. Unplug the power cord\n"
                "2. Plug/unplug minimal number of cables to only have the first "
                f"{i + 1} motor(s) ({self.motor_names[:i + 1]}) connected.\n"
                "3. Re-plug the power cord\n"
                "Press Enter to continue..."
            )
            print()
            self.reconnect()

            if i > 0:
                try:
                    self._read_with_motor_ids(self.motor_models, untaken_ids[:i], "ID")
                except ConnectionError:
                    print(f"Failed to read from {untaken_ids[:i + 1]}. Make sure the power cord is plugged in.")
                    input("Press Enter to continue...")
                    print()
                    self.reconnect()

            print("Scanning possible baudrates and motor indices")
            motor_found = False
            for baudrate in possible_baudrates:
                self.set_bus_baudrate(baudrate)
                present_ids = self.find_motor_indices(possible_ids)
                if len(present_ids) == 1:
                    present_idx = present_ids[0]
                    print(f"Detected motor with index {present_idx}")

                    if baudrate != BAUDRATE:
                        print(f"Setting its baudrate to {BAUDRATE}")
                        baudrate_idx = list(X_SERIES_BAUDRATE_TABLE.values()).index(BAUDRATE)

                        # The write can fail, so we allow retries
                        for _ in range(NUM_WRITE_RETRY):
                            self._write_with_motor_ids(
                                self.motor_models, present_idx, "Baud_Rate", baudrate_idx
                            )
                            time.sleep(0.5)
                            self.set_bus_baudrate(BAUDRATE)
                            try:
                                present_baudrate_idx = self._read_with_motor_ids(
                                    self.motor_models, present_idx, "Baud_Rate"
                                )
                            except ConnectionError:
                                print("Failed to write baudrate. Retrying.")
                                self.set_bus_baudrate(baudrate)
                                continue
                            break
                        else:
                            raise

                        if present_baudrate_idx != baudrate_idx:
                            raise OSError("Failed to write baudrate.")

                    print(f"Setting its index to a temporary untaken index ({untaken_ids[i]})")
                    self._write_with_motor_ids(self.motor_models, present_idx, "ID", untaken_ids[i])

                    present_idx = self._read_with_motor_ids(self.motor_models, untaken_ids[i], "ID")
                    if present_idx != untaken_ids[i]:
                        raise OSError("Failed to write index.")

                    motor_found = True
                    break
                elif len(present_ids) > 1:
                    raise OSError(f"More than one motor detected ({present_ids}), but only one was expected.")

            if not motor_found:
                raise OSError(
                    "No motor found, but one new motor expected. Verify power cord is plugged in and retry."
                )
            print()

        print(f"Setting expected motor indices: {self.motor_indices}")
        self.set_bus_baudrate(BAUDRATE)
        self._write_with_motor_ids(
            self.motor_models, untaken_ids[: len(self.motors)], "ID", self.motor_indices
        )
        print()

        if (self.read("ID") != self.motor_indices).any():
            raise OSError("Failed to write motors indices.")

        print("Configuration is done!")

    def find_motor_indices(self, possible_ids=None):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        indices = []
        for idx in tqdm.tqdm(possible_ids):
            try:
                present_idx = self._read_with_motor_ids(self.motor_models, [idx], "ID")[0]
            except ConnectionError:
                continue

            if idx != present_idx:
                # sanity check
                raise OSError(
                    "Motor index used to communicate through the bus is not the same as the one present in the motor memory. The motor memory might be damaged."
                )
            indices.append(idx)

        return indices

    def set_bus_baudrate(self, baudrate):
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise OSError("Failed to write bus baud rate.")

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, dynamixel xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32[ to centered signed int32 range [-2**31, 2**31[
        values = values.astype(np.int32)

        for i, name in enumerate(motor_names):
            homing_offset, drive_mode = self.calibration[name]

            # Update direction of rotation of the motor to match between leader and follower. In fact, the motor of the leader for a given joint
            # can be assembled in an opposite direction in term of rotation than the motor of the follower on the same joint.
            if drive_mode:
                values[i] *= -1

            # Convert from range [-2**31, 2**31[ to nominal range ]-resolution, resolution[ (e.g. ]-2048, 2048[)
            values[i] += homing_offset

        # Convert from range ]-resolution, resolution[ to the universal float32 centered degree range ]-180, 180[
        values = values.astype(np.float32)
        for i, name in enumerate(motor_names):
            _, model = self.motors[name]
            resolution = self.model_resolution[model]
            values[i] = values[i] / (resolution // 2) * 180

        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from the universal float32 centered degree range ]-180, 180[ to resolution range ]-resolution, resolution[
        for i, name in enumerate(motor_names):
            _, model = self.motors[name]
            resolution = self.model_resolution[model]
            values[i] = values[i] / 180 * (resolution // 2)

        values = np.round(values).astype(np.int32)

        # Convert from nominal range ]-resolution, resolution[ to centered signed int32 range [-2**31, 2**31[
        for i, name in enumerate(motor_names):
            homing_offset, drive_mode = self.calibration[name]
            values[i] -= homing_offset

            # Update direction of rotation of the motor that was matching between leader and follower to their original direction.
            # In fact, the motor of the leader for a given joint can be assembled in an opposite direction in term of rotation
            # than the motor of the follower on the same joint.
            if drive_mode:
                values[i] *= -1

        return values

    def _read_with_motor_ids(self, motor_models, motor_ids, data_name):
        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
        for idx in motor_ids:
            group.addParam(idx)

        comm = group.txRxPacket()
        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = group.getData(idx, addr, bytes)
            values.append(value)

        if return_list:
            return values
        else:
            return values[0]

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        if data_name not in self.group_readers:
            # create new group reader
            self.group_readers[group_key] = GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
            for idx in motor_ids:
                self.group_readers[group_key].addParam(idx)

        for _ in range(NUM_READ_RETRY):
            comm = self.group_readers[group_key].txRxPacket()
            if comm == COMM_SUCCESS:
                break

        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = self.group_readers[group_key].getData(idx, addr, bytes)
            values.append(value)

        values = np.array(values)

        # Convert to signed int to use range [-2048, 2048] for our motor positions.
        if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
            values = values.astype(np.int32)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration(values, motor_names)

            # We expect our motors to stay in a nominal range of [-180, 180] degrees
            # which corresponds to a half turn rotation.
            # However, some motors can turn a bit more, hence we extend the nominal range to [-270, 270]
            # which is less than a full 360 degree rotation.
            if not np.all((values > -270) & (values < 270)):
                raise ValueError(
                    f"Wrong motor position range detected. "
                    f"Expected to be in [-270, +270] but in [{values.min()}, {values.max()}]. "
                    "This might be due to a cable connection issue creating an artificial 360 degrees jump in motor values. "
                    "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                )

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def _write_with_motor_ids(self, motor_models, motor_ids, data_name, values):
        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        assert_same_address(self.model_ctrl_table, motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
            group.addParam(idx, data)

        comm = group.txPacket()
        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        values = values.tolist()

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        init_group = data_name not in self.group_readers
        if init_group:
            self.group_writers[group_key] = GroupSyncWrite(
                self.port_handler, self.packet_handler, addr, bytes
            )

        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
            if init_group:
                self.group_writers[group_key].addParam(idx, data)
            else:
                self.group_writers[group_key].changeParam(idx, data)

        comm = self.group_writers[group_key].txPacket()
        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler.closePort()
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
