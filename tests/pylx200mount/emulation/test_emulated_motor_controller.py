import asyncio
import contextlib
import logging
import math
import typing
import unittest.mock
from dataclasses import dataclass
from unittest import IsolatedAsyncioTestCase

import astropy.units as u
import pylx200mount
from astropy.coordinates import Angle


@dataclass
class ExpectedData:
    time: float
    position: int
    velocity: int


class TestEmulatedMotorController(IsolatedAsyncioTestCase):
    @contextlib.asynccontextmanager
    async def create_emulated_motor(
        self, initial_position: Angle = Angle(0.0, u.deg)
    ) -> typing.AsyncGenerator[None, None]:
        log = logging.getLogger(type(self).__name__)
        self.t = 0.0
        self.conversion_factor = Angle(0.0001 * u.deg)
        with unittest.mock.patch(
            "pylx200mount.emulation.emulated_motor_controller.DatetimeUtil.get_timestamp",
            self.get_timestamp,
        ):
            async with pylx200mount.emulation.EmulatedMotorController(
                initial_position=Angle(0.0, u.deg),
                log=log,
                conversion_factor=self.conversion_factor,
                hub_port=0,
            ) as self.emulated_motor_controller:
                self.emulated_motor_controller.position = initial_position
                self.emulated_motor_controller.stepper._position = self.emulated_motor_controller._position
                yield

    async def test_init_emulated_motor(self) -> None:
        log = logging.getLogger(type(self).__name__)
        conversion_factor = Angle(0.0001 * u.deg)
        emulated_motor_controller = pylx200mount.emulation.EmulatedMotorController(
            initial_position=Angle(0.0, u.deg),
            log=log,
            conversion_factor=conversion_factor,
            hub_port=0,
        )
        assert emulated_motor_controller.name == pylx200mount.motor.BaseMotorController.ALT
        assert not emulated_motor_controller.attached

        await emulated_motor_controller.connect()
        assert emulated_motor_controller.attached

        await emulated_motor_controller.disconnect()
        assert not emulated_motor_controller.attached

    def get_timestamp(self) -> float:
        return self.t

    async def assert_position_and_velocity(self, expected_data: ExpectedData) -> None:
        self.t = expected_data.time
        await asyncio.sleep(0.01)
        motor_pos_steps = (
            self.emulated_motor_controller.position / self.emulated_motor_controller._conversion_factor
        )
        motor_vel_steps = (
            self.emulated_motor_controller.velocity / self.emulated_motor_controller._conversion_factor
        )
        assert math.isclose(motor_pos_steps, expected_data.position, abs_tol=5), (
            f"{motor_pos_steps=}, {expected_data.position}"
        )
        assert math.isclose(motor_vel_steps, expected_data.velocity, abs_tol=5), (
            f"{motor_vel_steps=}, {expected_data.velocity}"
        )

    async def test_set_position(self) -> None:
        async with self.create_emulated_motor():
            assert math.isclose(self.emulated_motor_controller.position.deg, 0.0)
            assert math.isclose(self.emulated_motor_controller._position_offset, 0.0)

            self.emulated_motor_controller.position = Angle(1.0 * u.deg)
            assert math.isclose(self.emulated_motor_controller.position.deg, 1.0)
            assert math.isclose(self.emulated_motor_controller._position, 0.0)
            assert math.isclose(self.emulated_motor_controller._position_offset, 10000.0)

            self.emulated_motor_controller.position = Angle(2.0 * u.deg)
            assert math.isclose(self.emulated_motor_controller.position.deg, 2.0)
            assert math.isclose(self.emulated_motor_controller._position, 0.0)
            assert math.isclose(self.emulated_motor_controller._position_offset, 20000.0)

    async def test_move_far_positive(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 2.0, 100000, 100000),
                ExpectedData(command_time + 3.0, 200000, 100000),
                ExpectedData(command_time + 10.0, 900000, 100000),
                ExpectedData(command_time + 11.0, 975000, 50000),
                ExpectedData(command_time + 12.0, 1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_positive_from_position(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(45.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 475000, 50000),
                ExpectedData(command_time + 2.0, 550000, 100000),
                ExpectedData(command_time + 3.0, 650000, 100000),
                ExpectedData(command_time + 6.0, 943750, 75000),
                ExpectedData(command_time + 7.0, 993750, 25000),
                ExpectedData(command_time + 8.0, 1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_positive_to_same_position(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(45.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 475000, 50000),
                ExpectedData(command_time + 2.0, 550000, 100000),
                ExpectedData(command_time + 3.0, 650000, 100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 3.1, 660000, 100000),
                ExpectedData(command_time + 6.0, 943750, 75000),
                ExpectedData(command_time + 7.0, 993750, 25000),
                ExpectedData(command_time + 8.0, 1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_positive_to_different_position(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 2.0, 100000, 100000),
                ExpectedData(command_time + 3.0, 200000, 100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 3.1, 209750, 95000),
                ExpectedData(command_time + 4.0, 275000, 50000),
                ExpectedData(command_time + 5.0, 300000, 0),
                ExpectedData(command_time + 6.0, 275000, -50000),
                ExpectedData(command_time + 7.0, 200000, -100000),
                ExpectedData(command_time + 8.0, 125000, -50000),
                ExpectedData(command_time + 9.0, 100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_positive_to_different_position_from_pos(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(45.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1045000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 475000, 50000),
                ExpectedData(command_time + 2.0, 550000, 100000),
                ExpectedData(command_time + 3.0, 650000, 100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=145000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 3.1, 659750, 95000),
                ExpectedData(command_time + 4.0, 725000, 50000),
                ExpectedData(command_time + 5.0, 750000, 0),
                ExpectedData(command_time + 6.0, 725000, -50000),
                ExpectedData(command_time + 7.0, 650000, -100000),
                ExpectedData(command_time + 11.0, 250000, -100000),
                ExpectedData(command_time + 12.0, 172563, -52500),
                ExpectedData(command_time + 13.0, 145062, -2500),
                ExpectedData(command_time + 13.5, 145000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_positive_to_different_position_from_neg(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(-50.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=50000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, -475000, 50000),
                ExpectedData(command_time + 2.0, -400000, 100000),
                ExpectedData(command_time + 3.0, -300000, 100000),
                ExpectedData(command_time + 4.0, -200000, 100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=-250000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 4.1, -190250, 95000),
                ExpectedData(command_time + 5.0, -125000, 50000),
                ExpectedData(command_time + 6.0, -100000, 0),
                ExpectedData(command_time + 7.0, -125000, -50000),
                ExpectedData(command_time + 8.0, -196410, -73205),
                ExpectedData(command_time + 9.0, -244615, -23205),
                ExpectedData(command_time + 10.0, -250000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_positive_in_two_steps(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 2.0, 100000, 100000),
                ExpectedData(command_time + 10.0, 900000, 100000),
                ExpectedData(command_time + 11.0, 975000, 50000),
                ExpectedData(command_time + 12.0, 1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=2000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 13.0, 1025000, 50000),
                ExpectedData(command_time + 14.0, 1100000, 100000),
                # Angle wrap!!!
                ExpectedData(command_time + 22.0, -1700000, 100000),
                ExpectedData(command_time + 23.0, -1625000, 50000),
                ExpectedData(command_time + 24.0, -1600000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_near_positive(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 0.5, 6250, 25000),
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 1.5, 55882, 66421),
                ExpectedData(command_time + 2.0, 82843, 41421),
                ExpectedData(command_time + 2.5, 97303, 16421),
                ExpectedData(command_time + 2.9, 100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_near_positive_and_back(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 0.5, 6250, 25000),
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 1.5, 55882, 66421),
                ExpectedData(command_time + 2.0, 82843, 41421),
                ExpectedData(command_time + 2.5, 97303, 16421),
                ExpectedData(command_time + 3.0, 100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=0 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 3.5, 93750, -25000),
                ExpectedData(command_time + 4.0, 75000, -50000),
                ExpectedData(command_time + 4.5, 44118, -66421),
                ExpectedData(command_time + 5.0, 17157, -41421),
                ExpectedData(command_time + 5.5, 2697, -16421),
                ExpectedData(command_time + 6.0, 0, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(
                target_position=-1000000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 2.0, -100000, -100000),
                ExpectedData(command_time + 3.0, -200000, -100000),
                ExpectedData(command_time + 10.0, -900000, -100000),
                ExpectedData(command_time + 11.0, -975000, -50000),
                ExpectedData(command_time + 12.0, -1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative_from_position(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(45.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(
                target_position=-1000000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 1.0, 425000, -50000),
                ExpectedData(command_time + 2.0, 350000, -100000),
                ExpectedData(command_time + 3.0, 250000, -100000),
                ExpectedData(command_time + 10.0, -450000, -100000),
                ExpectedData(command_time + 11.0, -550000, -100000),
                ExpectedData(command_time + 12.5, -700000, -100000),
                ExpectedData(command_time + 15.0, -943750, -75000),
                ExpectedData(command_time + 16.0, -993750, -25000),
                ExpectedData(command_time + 16.5, -1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative_to_same_position(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(45.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(
                target_position=-1000000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 1.0, 425000, -50000),
                ExpectedData(command_time + 2.0, 350000, -100000),
                ExpectedData(command_time + 3.0, 250000, -100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(
                target_position=-1000000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 3.1, 240000, -100000),
                ExpectedData(command_time + 10.0, -450000, -100000),
                ExpectedData(command_time + 11.0, -550000, -100000),
                ExpectedData(command_time + 12.5, -700000, -100000),
                ExpectedData(command_time + 15.0, -943750, -75000),
                ExpectedData(command_time + 16.0, -993750, -25000),
                ExpectedData(command_time + 16.5, -1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative_to_different_position(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(
                target_position=-1000000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 2.0, -100000, -100000),
                ExpectedData(command_time + 3.0, -200000, -100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(
                target_position=-100000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 3.1, -209750, -95000),
                ExpectedData(command_time + 4.0, -275000, -50000),
                ExpectedData(command_time + 5.0, -300000, 0),
                ExpectedData(command_time + 6.0, -275000, 50000),
                ExpectedData(command_time + 7.0, -200000, 100000),
                ExpectedData(command_time + 8.0, -125000, 50000),
                ExpectedData(command_time + 9.0, -100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative_to_different_position_from_pos(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(45.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(
                target_position=-955000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 1.0, 425000, -50000),
                ExpectedData(command_time + 2.0, 350000, -100000),
                ExpectedData(command_time + 3.0, 250000, -100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(
                target_position=550000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 3.1, 240250, -95000),
                ExpectedData(command_time + 4.0, 175000, -50000),
                ExpectedData(command_time + 5.0, 150000, 0),
                ExpectedData(command_time + 6.0, 175000, 50000),
                ExpectedData(command_time + 7.0, 250000, 100000),
                ExpectedData(command_time + 8.0, 350000, 100000),
                ExpectedData(command_time + 9.0, 450000, 100000),
                ExpectedData(command_time + 10.0, 525000, 50000),
                ExpectedData(command_time + 11.0, 550000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative_to_different_position_from_neg(self) -> None:
        async with self.create_emulated_motor(initial_position=Angle(-50.0, u.deg)):
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(
                target_position=-1050000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 1.0, -525000, -50000),
                ExpectedData(command_time + 2.0, -600000, -100000),
                ExpectedData(command_time + 3.0, -700000, -100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(
                target_position=-150000 * self.conversion_factor,
            )
            for expected_data in [
                ExpectedData(command_time + 3.1, -709750, -95000),
                ExpectedData(command_time + 4.0, -775000, -50000),
                ExpectedData(command_time + 5.0, -800000, 0),
                ExpectedData(command_time + 6.0, -775000, 50000),
                ExpectedData(command_time + 7.0, -700000, 100000),
                ExpectedData(command_time + 9.0, -500000, 100000),
                ExpectedData(command_time + 11.0, -300000, 100000),
                ExpectedData(command_time + 12.0, -206250, 75000),
                ExpectedData(command_time + 13.0, -156250, 25000),
                ExpectedData(command_time + 13.5, -150000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_far_negative_in_two_steps(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=-1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 2.0, -100000, -100000),
                ExpectedData(command_time + 10.0, -900000, -100000),
                ExpectedData(command_time + 11.0, -975000, -50000),
                ExpectedData(command_time + 12.0, -1000000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=-2000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 13.0, -1025000, -50000),
                ExpectedData(command_time + 14.0, -1100000, -100000),
                ExpectedData(command_time + 22.0, 1700000, -100000),
                ExpectedData(command_time + 23.0, 1625000, -50000),
                ExpectedData(command_time + 24.0, 1600000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_near_negative(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=-100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 0.5, -6250, -25000),
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 1.5, -55882, -66421),
                ExpectedData(command_time + 2.0, -82843, -41421),
                ExpectedData(command_time + 2.5, -97303, -16421),
                ExpectedData(command_time + 3.0, -100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_move_near_negative_and_back(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=-100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 0.5, -6250, -25000),
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 1.5, -55882, -66421),
                ExpectedData(command_time + 2.0, -82843, -41421),
                ExpectedData(command_time + 2.5, -97303, -16421),
                ExpectedData(command_time + 3.0, -100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.move(target_position=0 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 3.5, -93750, 25000),
                ExpectedData(command_time + 4.0, -75000, 50000),
                ExpectedData(command_time + 4.5, -44118, 66421),
                ExpectedData(command_time + 5.0, -17157, 41421),
                ExpectedData(command_time + 5.5, -2697, 16421),
                ExpectedData(command_time + 6.0, 0, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_pos_stop_while_at_max_speed(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 2.0, 100000, 100000),
                ExpectedData(command_time + 3.0, 200000, 100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.stop_motion()
            for expected_data in [
                ExpectedData(command_time + 3.1, 209750, 95000),
                ExpectedData(command_time + 4.0, 275000, 50000),
                ExpectedData(command_time + 5.0, 300000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_pos_stop_while_speeding_up(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 1.5, 56250, 75000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.stop_motion()
            for expected_data in [
                ExpectedData(command_time + 1.6, 63500, 70000),
                ExpectedData(command_time + 1.7, 70250, 65000),
                ExpectedData(command_time + 2.0, 87500, 50000),
                ExpectedData(command_time + 2.5, 106250, 25000),
                ExpectedData(command_time + 3.0, 112500, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_pos_stop_while_slowing_down(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 0.5, 6250, 25000),
                ExpectedData(command_time + 1.0, 25000, 50000),
                ExpectedData(command_time + 1.5, 55882, 66421),
                ExpectedData(command_time + 2.0, 82843, 41421),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.stop_motion()
            for expected_data in [
                ExpectedData(command_time + 2.5, 97303, 16421),
                ExpectedData(command_time + 3.0, 100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_neg_stop_while_at_max_speed(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=-1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 2.0, -100000, -100000),
                ExpectedData(command_time + 3.0, -200000, -100000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.stop_motion()
            for expected_data in [
                ExpectedData(command_time + 3.1, -209750, -95000),
                ExpectedData(command_time + 4.0, -275000, -50000),
                ExpectedData(command_time + 5.0, -300000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_neg_stop_while_speeding_up(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=-1000000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 1.5, -56250, -75000),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.stop_motion()
            for expected_data in [
                ExpectedData(command_time + 1.6, -63500, -70000),
                ExpectedData(command_time + 1.7, -70250, -65000),
                ExpectedData(command_time + 2.0, -87500, -50000),
                ExpectedData(command_time + 2.5, -106250, -25000),
                ExpectedData(command_time + 3.0, -112500, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_neg_stop_while_slowing_down(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.move(target_position=-100000 * self.conversion_factor)
            for expected_data in [
                ExpectedData(command_time + 0.5, -6250, -25000),
                ExpectedData(command_time + 1.0, -25000, -50000),
                ExpectedData(command_time + 1.5, -55882, -66421),
                ExpectedData(command_time + 2.0, -82843, -41421),
            ]:
                await self.assert_position_and_velocity(expected_data)
            await self.emulated_motor_controller.stop_motion()
            for expected_data in [
                ExpectedData(command_time + 2.5, -97303, -16421),
                ExpectedData(command_time + 3.0, -100000, 0),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_track(self) -> None:
        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.track(
                target_position=1000 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, 0, 0),
                ExpectedData(command_time + 0.5, 490, 1000),
                ExpectedData(command_time + 1.0, 990, 1000),
                ExpectedData(command_time + 1.5, 1490, 1000),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=1500 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, 490, 1000),
                ExpectedData(command_time + 0.5, 995, 1010),
                ExpectedData(command_time + 1.0, 1500, 1010),
                ExpectedData(command_time + 1.5, 2005, 1010),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=2000 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, 995, 1010),
                ExpectedData(command_time + 0.5, 1500, 1005),
                ExpectedData(command_time + 1.0, 2000, 1005),
                ExpectedData(command_time + 1.5, 2500, 1005),
            ]:
                await self.assert_position_and_velocity(expected_data)

        async with self.create_emulated_motor():
            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.track(
                target_position=-1000 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time + 0.5, -490, -1000),
                ExpectedData(command_time + 1.0, -990, -1000),
                ExpectedData(command_time + 1.5, -1490, -1000),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=-1500 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, -490, -1000),
                ExpectedData(command_time + 0.5, -995, -1010),
                ExpectedData(command_time + 1.0, -1500, -1010),
                ExpectedData(command_time + 1.5, -2005, -1010),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=-2000 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, -995, -1005),
                ExpectedData(command_time + 0.5, -1500, -1005),
                ExpectedData(command_time + 1.0, -2000, -1005),
                ExpectedData(command_time + 1.5, -2500, -1005),
            ]:
                await self.assert_position_and_velocity(expected_data)

    async def test_track_from_position(self) -> None:
        async with self.create_emulated_motor():
            self.emulated_motor_controller.position = 100 * self.conversion_factor

            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.track(
                target_position=1100 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, 100, 0),
                ExpectedData(command_time + 0.5, 590, 1000),
                ExpectedData(command_time + 1.0, 1090, 1000),
                ExpectedData(command_time + 1.5, 1590, 1000),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=1600 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, 590, 1000),
                ExpectedData(command_time + 0.5, 1095, 1010),
                ExpectedData(command_time + 1.0, 1600, 1010),
                ExpectedData(command_time + 1.5, 2105, 1010),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=2100 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, 1095, 1010),
                ExpectedData(command_time + 0.5, 1600, 1005),
                ExpectedData(command_time + 1.0, 2100, 1005),
                ExpectedData(command_time + 1.5, 2600, 1005),
            ]:
                await self.assert_position_and_velocity(expected_data)

        async with self.create_emulated_motor():
            self.emulated_motor_controller.position = -100 * self.conversion_factor

            command_time = self.t = pylx200mount.DatetimeUtil.get_timestamp()
            await self.emulated_motor_controller.track(
                target_position=-1100 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, -100, 0),
                ExpectedData(command_time + 0.5, -590, -1000),
                ExpectedData(command_time + 1.0, -1090, -1000),
                ExpectedData(command_time + 1.5, -1590, -1000),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=-1600 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, -590, -1000),
                ExpectedData(command_time + 0.5, -1095, -1010),
                ExpectedData(command_time + 1.0, -1600, -1010),
                ExpectedData(command_time + 1.5, -2105, -1010),
            ]:
                await self.assert_position_and_velocity(expected_data)

            command_time = self.t = command_time + 0.5
            await self.emulated_motor_controller.track(
                target_position=-2100 * self.conversion_factor, timediff=1.0
            )
            for expected_data in [
                ExpectedData(command_time, -1095, -1010),
                ExpectedData(command_time + 0.5, -1600, -1005),
                ExpectedData(command_time + 1.0, -2100, -1005),
                ExpectedData(command_time + 1.5, -2600, -1005),
            ]:
                await self.assert_position_and_velocity(expected_data)
