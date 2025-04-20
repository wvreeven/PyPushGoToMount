import logging
import pathlib
from typing import Tuple
from unittest import IsolatedAsyncioTestCase, mock

import astropy.units as u
import pylx200mount
import pytest
from astropy.coordinates import SkyCoord

CONFIG_DIR = pathlib.Path(__file__).parents[1] / "test_data"


def format_ra_dec_str(ra_dec: SkyCoord) -> Tuple[str, str]:
    ra = ra_dec.ra
    ra_str = ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
    dec = ra_dec.dec
    dec_dms = dec.signed_dms
    dec_str = f"{dec_dms.sign * dec_dms.d:2.0f}*{dec_dms.m:2.0f}:{dec_dms.s:2.0f}"
    return ra_str, dec_str


class TestMountController(IsolatedAsyncioTestCase):
    async def get_ra_dec_str_from_alt_az(self, alt: float, az: float) -> Tuple[str, str]:
        alt_az = await pylx200mount.my_math.get_skycoord_from_alt_az(
            alt=alt,
            az=az,
            timestamp=pylx200mount.DatetimeUtil.get_timestamp(),
        )
        ra_dec = await pylx200mount.my_math.get_radec_from_altaz(alt_az=alt_az)
        return format_ra_dec_str(ra_dec)

    async def async_set_up(self) -> None:
        log = logging.getLogger(type(self).__name__)
        self.mount_controller = pylx200mount.controller.MountController(log=log)
        ra_str, dec_str = await self.get_ra_dec_str_from_alt_az(alt=45.0, az=175.0)
        self.target_ra_str, self.target_dec_str = await self.get_ra_dec_str_from_alt_az(alt=40.0, az=179.0)
        ra_dec = await pylx200mount.my_math.get_skycoord_from_ra_dec_str(ra_str=ra_str, dec_str=dec_str)
        await self.mount_controller.start()
        await self.mount_controller.set_ra_dec(ra_dec=ra_dec)

    async def asyncTearDown(self) -> None:
        await self.mount_controller.stop()

    async def test_slew_to_altaz(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_motors_only.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            assert self.mount_controller.motor_controller_alt is not None
            assert self.mount_controller.motor_controller_az is not None
            assert self.mount_controller.plate_solver is None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.MOTORS_ONLY

            slew_to = await self.mount_controller.slew_to(
                ra_str=self.target_ra_str, dec_str=self.target_dec_str
            )
            assert "0" == slew_to
            self.target_ra_str, self.target_dec_str = await self.get_ra_dec_str_from_alt_az(
                alt=-40.0, az=-179.0
            )
            slew_to = await self.mount_controller.slew_to(
                ra_str=self.target_ra_str, dec_str=self.target_dec_str
            )
            assert "1" == slew_to

    async def test_slew_bad(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_motors_only.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            assert self.mount_controller.motor_controller_alt is not None
            assert self.mount_controller.motor_controller_az is not None
            assert self.mount_controller.plate_solver is None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.MOTORS_ONLY

            bad_mode = "Bad"
            self.mount_controller.slew_mode = bad_mode  # type: ignore
            slew_to = await self.mount_controller.slew_to(
                ra_str=self.target_ra_str, dec_str=self.target_dec_str
            )
            assert "0" == slew_to

    async def test_slew_in_direction(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_motors_only.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            assert self.mount_controller.motor_controller_alt is not None
            assert self.mount_controller.motor_controller_az is not None
            assert self.mount_controller.plate_solver is None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.MOTORS_ONLY

            await self.mount_controller.slew_in_direction("Mn")
            assert self.mount_controller.slew_direction == pylx200mount.SlewDirection.UP
            await self.mount_controller.slew_in_direction("Me")
            assert self.mount_controller.slew_direction == pylx200mount.SlewDirection.LEFT
            await self.mount_controller.slew_in_direction("Ms")
            assert self.mount_controller.slew_direction == pylx200mount.SlewDirection.DOWN
            await self.mount_controller.slew_in_direction("Mw")
            assert self.mount_controller.slew_direction == pylx200mount.SlewDirection.RIGHT
            with pytest.raises(ValueError):
                await self.mount_controller.slew_in_direction("MM")
            await self.mount_controller.stop_slew()
            assert self.mount_controller.slew_direction == pylx200mount.SlewDirection.NONE

    async def test_set_slew_rate(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_motors_only.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            assert self.mount_controller.motor_controller_alt is not None
            assert self.mount_controller.motor_controller_az is not None
            assert self.mount_controller.plate_solver is None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.MOTORS_ONLY

            await self.mount_controller.set_slew_rate("RC")
            assert self.mount_controller.slew_rate == pylx200mount.SlewRate.CENTERING
            await self.mount_controller.set_slew_rate("RG")
            assert self.mount_controller.slew_rate == pylx200mount.SlewRate.GUIDING
            await self.mount_controller.set_slew_rate("RM")
            assert self.mount_controller.slew_rate == pylx200mount.SlewRate.FIND
            await self.mount_controller.set_slew_rate("RS")
            assert self.mount_controller.slew_rate == pylx200mount.SlewRate.HIGH
            with pytest.raises(ValueError):
                await self.mount_controller.set_slew_rate("RR")

    async def test_with_emulated_camera(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_camera_only.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            self.mount_controller.plate_solver.solve = self.solve  # type: ignore
            assert self.mount_controller.motor_controller_alt is None
            assert self.mount_controller.motor_controller_az is None
            assert self.mount_controller.plate_solver is not None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.CAMERA_ONLY
            await self.mount_controller.stop()

    async def test_with_emulated_camera_and_motors(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_camera_and_motors.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            assert self.mount_controller.motor_controller_alt is not None
            assert self.mount_controller.motor_controller_az is not None
            assert self.mount_controller.plate_solver is not None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.CAMERA_AND_MOTORS
            await self.mount_controller.stop()

    async def test_empty(self) -> None:
        self.config_file = CONFIG_DIR / "config_empty.json"
        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            await self.async_set_up()
            assert self.mount_controller.motor_controller_alt is None
            assert self.mount_controller.motor_controller_az is None
            assert self.mount_controller.plate_solver is None
            assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.NONE

    async def test_track(self) -> None:
        self.config_file = CONFIG_DIR / "config_emulated_motors_only.json"
        with mock.patch(
            "pylx200mount.emulation.emulated_motor_controller.DatetimeUtil.get_timestamp",
            self.get_timestamp,
        ), mock.patch("pylx200mount.controller.utils.CONFIG_FILE", self.config_file):
            log = logging.getLogger(type(self).__name__)
            for az_offset in [0.0, -1.0]:
                self.t = 1000.0
                self.mount_controller = pylx200mount.controller.MountController(log=log)
                await self.mount_controller.start()
                await self.mount_controller.stop()

                assert self.mount_controller.motor_controller_alt is not None
                assert self.mount_controller.motor_controller_az is not None
                assert self.mount_controller.plate_solver is None
                assert self.mount_controller.controller_type == pylx200mount.MotorControllerType.MOTORS_ONLY

                await self.set_motor_controller_position(
                    alt=10.0, az=10.0, alt_offset=0.0, az_offset=az_offset
                )
                await self.set_motor_controller_position(
                    alt=10.0, az=70.0, alt_offset=0.0, az_offset=az_offset
                )
                await self.set_motor_controller_position(
                    alt=10.0, az=130.0, alt_offset=0.0, az_offset=az_offset
                )
                await self.deterine_motor_controller_position()

                for _ in range(10):
                    self.t += pylx200mount.controller.POSITION_INTERVAL
                    await self.deterine_motor_controller_position()

    def get_timestamp(self) -> float:
        return self.t

    async def set_motor_controller_position(
        self, alt: float, az: float, alt_offset: float, az_offset: float
    ) -> None:
        assert isinstance(
            self.mount_controller.motor_controller_alt, pylx200mount.emulation.EmulatedMotorController
        )
        assert isinstance(
            self.mount_controller.motor_controller_az, pylx200mount.emulation.EmulatedMotorController
        )
        alt_position_in_steps = (
            (alt - alt_offset) / self.mount_controller.motor_controller_alt._conversion_factor
        ).value
        self.mount_controller.motor_controller_alt._position = alt_position_in_steps
        self.mount_controller.motor_controller_alt.stepper._position = alt_position_in_steps
        self.mount_controller.motor_controller_alt.state = pylx200mount.MotorControllerState.SLEWING

        az_position_in_steps = (
            (az - az_offset) / self.mount_controller.motor_controller_az._conversion_factor
        ).value
        self.mount_controller.motor_controller_az._position = az_position_in_steps
        self.mount_controller.motor_controller_az.stepper._position = az_position_in_steps
        self.mount_controller.motor_controller_az.state = pylx200mount.MotorControllerState.SLEWING

        altaz = await pylx200mount.my_math.get_skycoord_from_alt_az(
            alt=alt, az=az, timestamp=pylx200mount.DatetimeUtil.get_timestamp()
        )
        radec = await pylx200mount.my_math.get_radec_from_altaz(altaz)
        await self.mount_controller.set_ra_dec(radec)

    async def deterine_motor_controller_position(self) -> None:
        assert isinstance(
            self.mount_controller.motor_controller_alt, pylx200mount.emulation.EmulatedMotorController
        )
        assert isinstance(
            self.mount_controller.motor_controller_az, pylx200mount.emulation.EmulatedMotorController
        )
        self.mount_controller.motor_controller_alt.stepper._compute_position_and_velocity()
        self.mount_controller.motor_controller_az.stepper._compute_position_and_velocity()
        await self.mount_controller.get_motor_positions()
        logging.getLogger(type(self).__name__).debug(
            f"{self.mount_controller.motor_controller_alt.stepper._trajectory}"
        )

    async def solve(self) -> SkyCoord:
        return await pylx200mount.my_math.get_skycoord_from_ra_dec_str(
            self.target_ra_str, self.target_dec_str
        )
