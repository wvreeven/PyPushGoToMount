from __future__ import annotations

__all__ = ["POSITION_INTERVAL", "MountController"]

import asyncio
import importlib
import logging
import types

from astropy import units as u
from astropy.coordinates import AltAz, Angle, SkyCoord

from ..alignment import AlignmentHandler, TelescopeAltAzFrame
from ..camera import BaseCamera
from ..datetime_util import DatetimeUtil
from ..enums import MILLISECOND, MotorControllerState, MotorControllerType, SlewDirection, SlewRate
from ..motor.base_motor_controller import BaseMotorController
from ..my_math.astropy_util import (
    get_altaz_from_radec,
    get_radec_from_altaz,
    get_skycoord_from_alt_az,
    get_skycoord_from_ra_dec_str,
)
from ..observing_location import get_observing_location
from ..plate_solver import BasePlateSolver
from .utils import load_config

# Angle of 90º.
NINETY = Angle(90.0, u.deg)
# Angle of 0º.
ZERO = Angle(0.0, u.deg)
# Position loop task interval [sec].
POSITION_INTERVAL = 0.25


class MountController:
    """Control the Mount."""

    def __init__(self, log: logging.Logger) -> None:
        self.log = log.getChild(type(self).__name__)

        self.configuration: types.SimpleNamespace | None = None

        self.controller_type = MotorControllerType.NONE

        # The motor controllers.
        self.motor_controller_alt: BaseMotorController | None = None
        self.motor_controller_az: BaseMotorController | None = None

        # Position loop that is done, so it can be safely canceled at all times.
        self._position_loop_task: asyncio.Future = asyncio.Future()
        self._position_loop_task.set_result(None)
        self.should_run_position_loop = False
        self.motor_alt_az: SkyCoord | None = None

        # Target RaDec for moves and tracking.
        self.target_radec = SkyCoord(0.0 * u.deg, 0.0 * u.deg)

        # Position event to set in the position loop. Used by unit tests.
        self.position_event: asyncio.Event = asyncio.Event()

        # Slew related variables.
        self.slew_direction = SlewDirection.NONE
        self.slew_rate = SlewRate.HIGH

        # The plate solver.
        self.plate_solver: BasePlateSolver | None = None

        # Plate solve loop that is done, so it can be safely canceled at all times.
        self._plate_solve_loop_task: asyncio.Future = asyncio.Future()
        self._plate_solve_loop_task.set_result(None)
        self.should_run_plate_solve_loop = False
        self.camera_alt_az: SkyCoord | None = None
        self.previous_camera_alt_az: SkyCoord | None = None

        # Alignment handler.
        self.camera_alignment_handler = AlignmentHandler()

    async def load_motors_camera_and_plate_solver(self) -> None:
        """Helper method to load the configured motors, camera and plate solver."""
        zero_alt_az = await get_skycoord_from_alt_az(
            alt=0.0,
            az=0.0,
            timestamp=DatetimeUtil.get_timestamp(),
            frame=TelescopeAltAzFrame,
        )
        self.motor_alt_az = zero_alt_az
        self.camera_alt_az = zero_alt_az
        self.previous_camera_alt_az = zero_alt_az

        self.configuration = load_config()
        assert self.configuration is not None

        if hasattr(self.configuration, "alt_module_name"):
            self.log.debug(
                f"Loading ALT motor {self.configuration.alt_module_name}.{self.configuration.alt_class_name}."
            )
            self.log.debug(
                f"Loading AZ motor {self.configuration.az_module_name}.{self.configuration.az_class_name}."
            )
            alt_motor_module = importlib.import_module(self.configuration.alt_module_name)
            alt_motor_class = getattr(alt_motor_module, self.configuration.alt_class_name)
            az_motor_module = importlib.import_module(self.configuration.az_module_name)
            az_motor_class = getattr(az_motor_module, self.configuration.az_class_name)

            # The motor controllers.
            self.motor_controller_alt = alt_motor_class(
                initial_position=Angle(0.0, u.deg),
                log=self.log,
                conversion_factor=Angle(self.configuration.alt_gear_reduction * u.deg),
                hub_port=self.configuration.alt_hub_port,
            )
            self.motor_controller_az = az_motor_class(
                initial_position=Angle(0.0, u.deg),
                log=self.log,
                conversion_factor=Angle(self.configuration.az_gear_reduction * u.deg),
                hub_port=self.configuration.az_hub_port,
            )
        else:
            self.log.warning("No motors connected.")

        if hasattr(self.configuration, "camera_module_name"):
            self.log.debug(
                f"Loading camera "
                f"{self.configuration.camera_module_name}.{self.configuration.camera_class_name}."
            )
            if self.configuration.camera_class_name == "EmulatedCamera":
                from ..emulation import EmulatedCamera, EmulatedPlateSolver

                camera: BaseCamera = EmulatedCamera(log=self.log)
                self.plate_solver = EmulatedPlateSolver(
                    camera,
                    self.configuration.camera_focal_length,
                    self.log,
                )
            else:
                from ..plate_solver import PlateSolver

                camera_module = importlib.import_module(self.configuration.camera_module_name)
                camera_class = getattr(camera_module, self.configuration.camera_class_name)
                camera = camera_class(log=self.log)
                self.plate_solver = PlateSolver(
                    camera,
                    self.configuration.camera_focal_length,
                    self.log,
                )
        else:
            self.log.debug("No camera connected.")

        if self.motor_controller_alt and self.motor_controller_az:
            if self.plate_solver:
                self.controller_type = MotorControllerType.CAMERA_AND_MOTORS
            else:
                self.controller_type = MotorControllerType.MOTORS_ONLY
        elif self.plate_solver:
            self.controller_type = MotorControllerType.CAMERA_ONLY
        self.log.debug(f"{self.controller_type=}")

    async def start(self) -> None:
        """Start the mount controller.

        The main actions are to start the position loop, to connect the motors and to perform other start up
        actions.
        """
        self.log.info("Start called.")
        await self.load_motors_camera_and_plate_solver()
        await self.attach_motors()
        await self.start_plate_solver()
        self.log.info("Started.")

    async def start_plate_solver(self) -> None:
        """Let the camera start taking images and start the plate solve task."""
        if self.controller_type in [
            MotorControllerType.CAMERA_ONLY,
            MotorControllerType.CAMERA_AND_MOTORS,
        ]:
            assert self.plate_solver is not None
            try:
                self.log.debug("Starting plate solver.")
                await self.plate_solver.open_camera()
                await self.plate_solver.start_imaging()

                self.should_run_plate_solve_loop = True
                self._plate_solve_loop_task.cancel()
                await self._plate_solve_loop_task
                self._plate_solve_loop_task = asyncio.create_task(self.plate_solve_loop())
                self.log.debug("Plate solver started.")
            except Exception:
                self.log.exception("Error loading configured camera. Continuing without camera.")
                self.should_run_plate_solve_loop = False
                self.plate_solver = None
                if self.controller_type == MotorControllerType.CAMERA_AND_MOTORS:
                    self.controller_type = MotorControllerType.MOTORS_ONLY
                else:
                    self.controller_type = MotorControllerType.NONE

    async def attach_motors(self) -> None:
        """Attach the motors."""
        if self.motor_controller_alt is None or self.motor_controller_az is None:
            if self.controller_type == MotorControllerType.CAMERA_AND_MOTORS:
                self.controller_type = MotorControllerType.CAMERA_ONLY
            elif self.controller_type == MotorControllerType.MOTORS_ONLY:
                self.controller_type = MotorControllerType.NONE
            return

        await self.motor_controller_alt.connect()
        await self.motor_controller_az.connect()

        self.should_run_position_loop = True
        self._position_loop_task.cancel()
        await self._position_loop_task
        self._position_loop_task = asyncio.create_task(self.position_loop())

    async def position_loop(self) -> None:
        """The position loop.

        Get the motor positions every `POSITION_INTERVAL` seconds and let the motors track if necessary. The
        loop delay is non-drifiting.
        """
        start_time = DatetimeUtil.get_timestamp()
        self.log.debug(f"position_loop starts at {start_time}")
        while self.should_run_position_loop:
            await self.get_motor_positions()

            remainder = (DatetimeUtil.get_timestamp() - start_time) % POSITION_INTERVAL
            await asyncio.sleep(POSITION_INTERVAL - remainder)

    async def get_motor_positions(self) -> None:
        assert self.motor_controller_alt is not None
        assert self.motor_controller_az is not None

        self.log.debug("Getting motor positions.")
        self.motor_alt_az = await get_skycoord_from_alt_az(
            alt=self.motor_controller_alt.position.deg,
            az=self.motor_controller_az.position.deg,
            timestamp=DatetimeUtil.get_timestamp(),
            frame=TelescopeAltAzFrame,
        )
        self.position_event.set()

        self.check_motor_tracking(self.motor_controller_az)
        self.check_motor_tracking(self.motor_controller_alt)
        # Since the slew is performed to the AltAz at the end of the longest axis slew, tracking the position
        # should only start as soon as both motors are in TRACKING state.
        if (
            self.motor_controller_az.state == MotorControllerState.TRACKING
            and self.motor_controller_alt.state == MotorControllerState.TRACKING
        ):
            fut_timestamp = DatetimeUtil.get_timestamp() + POSITION_INTERVAL
            target_alt_az = await get_altaz_from_radec(self.target_radec, fut_timestamp)

            await self.motor_controller_az.track(target_alt_az.az, POSITION_INTERVAL)
            await self.motor_controller_alt.track(target_alt_az.alt, POSITION_INTERVAL)

    def check_motor_tracking(self, motor: BaseMotorController) -> None:
        """Check if the provided motor is stopped.

        If the motor state is not stopped but the motor velocity is 0 deg/sec, then the motor state is set to
        `MotorControllerState.TRACKING`.

        Parameters
        ----------
        motor : `BaseMotorController`
            The motor to check.
        """
        if motor.state != MotorControllerState.STOPPED and motor.velocity == ZERO:
            motor.state = MotorControllerState.TRACKING

    async def stop(self) -> None:
        """Stop the mount controller.

        The main actions are to stop the position loop, to disconnect the motors and to perform other shut
        down actions.
        """
        self.log.info("Stop called.")
        await self.stop_plate_solver()
        await self.detach_motors()
        self.log.info("Stopped.")

    async def stop_plate_solver(self) -> None:
        """Stop the plate solve task and let the camera stop taking images."""
        if self.controller_type in [
            MotorControllerType.CAMERA_AND_MOTORS,
            MotorControllerType.CAMERA_ONLY,
        ]:
            self.log.debug(f"{self.controller_type=}")
            assert self.plate_solver is not None
            self.should_run_plate_solve_loop = False
            await self._plate_solve_loop_task
            await self.plate_solver.stop_imaging()

    async def detach_motors(self) -> None:
        """Detach the motors and stop the position loop."""
        if self.motor_controller_alt is None or self.motor_controller_az is None:
            if self.controller_type == MotorControllerType.CAMERA_AND_MOTORS:
                self.controller_type = MotorControllerType.CAMERA_ONLY
            elif self.controller_type == MotorControllerType.MOTORS_ONLY:
                self.controller_type = MotorControllerType.NONE
            return

        self.should_run_position_loop = False
        await self._position_loop_task

        await self.motor_controller_alt.disconnect()
        await self.motor_controller_az.disconnect()

    async def plate_solve_loop(self) -> None:
        start_time = DatetimeUtil.get_timestamp()
        self.log.debug(f"plate_solve_loop starts at {start_time}")
        while self.should_run_plate_solve_loop:
            await self.perform_plate_solve()

    async def perform_plate_solve(self) -> None:
        assert self.plate_solver is not None
        now = DatetimeUtil.get_timestamp()
        try:
            assert self.camera_alt_az is not None
            self.previous_camera_alt_az = self.camera_alt_az
            camera_ra_dec = await self.plate_solver.solve()
            self.camera_alt_az = await get_altaz_from_radec(
                ra_dec=camera_ra_dec, timestamp=now, frame=TelescopeAltAzFrame
            )
            # Make sure that the motors know the camera position as well.
            if self.controller_type == MotorControllerType.CAMERA_AND_MOTORS:
                assert self.motor_controller_alt is not None
                assert self.motor_controller_az is not None
                self.motor_controller_alt.position = self.camera_alt_az.alt
                self.motor_controller_az.position = self.camera_alt_az.az

            self.log.debug("Camera RaDec = %s", camera_ra_dec.to_string("hmsdms"))
            self.log.debug("Camera AltAz = %s", self.camera_alt_az.to_string("dms"))

        except RuntimeError:
            self.log.exception("Error solving.")
            assert self.previous_camera_alt_az is not None
            self.camera_alt_az = self.previous_camera_alt_az
        end = DatetimeUtil.get_timestamp()
        self.log.debug(f"Plate solve for mount AltAz took {end - now} s.")

    async def get_ra_dec(self) -> SkyCoord:
        """Get the current RA and DEC of the mount.

        Since RA and DEC of the mount are requested in pairs, this method computes both
        the RA and DEC.

        Returns
        -------
        The right ascention and declination.
        """
        alignment_handler: AlignmentHandler | None = None
        match self.controller_type:
            case MotorControllerType.CAMERA_ONLY:
                assert self.camera_alt_az is not None
                mount_alt_az = self.camera_alt_az
                alignment_handler = self.camera_alignment_handler
            case MotorControllerType.MOTORS_ONLY:
                assert self.motor_alt_az is not None
                mount_alt_az = self.motor_alt_az
                alignment_handler = None
            case MotorControllerType.CAMERA_AND_MOTORS:
                assert self.motor_controller_alt is not None
                assert self.motor_controller_az is not None
                if (
                    self.motor_controller_az.state == MotorControllerState.SLEWING
                    or self.motor_controller_alt.state == MotorControllerState.SLEWING
                ):
                    mount_alt_az = self.motor_alt_az
                    alignment_handler = None
                else:
                    assert self.camera_alt_az is not None
                    mount_alt_az = self.camera_alt_az
                    alignment_handler = self.camera_alignment_handler
            case _:
                mount_alt_az = await get_skycoord_from_alt_az(
                    alt=0.0,
                    az=0.0,
                    timestamp=DatetimeUtil.get_timestamp(),
                    frame=TelescopeAltAzFrame,
                )

        if alignment_handler is not None:
            transformed_mount_alt_az = await alignment_handler.get_altaz_from_telescope_coords(mount_alt_az)
        else:
            transformed_mount_alt_az = mount_alt_az

        ra_dec = await get_radec_from_altaz(alt_az=transformed_mount_alt_az)
        return ra_dec

    async def set_ra_dec(self, ra_dec: SkyCoord) -> None:
        """Set the current RA and DEC of the mount.

        In case the mount has not been aligned yet, the AzAlt rotated frame of the
        mount gets calculated as well.

        Parameters
        ----------
        ra_dec: `SkyCoord`
            The RA and Dec of the mount.
        """
        now = DatetimeUtil.get_timestamp()
        self.target_radec = ra_dec

        # Determine the sky AltAz.
        sky_alt_az = await get_altaz_from_radec(ra_dec, now)

        if self.controller_type in [MotorControllerType.CAMERA_ONLY, MotorControllerType.CAMERA_AND_MOTORS]:
            assert self.camera_alt_az is not None
            camera_alt_az = await get_skycoord_from_alt_az(
                self.camera_alt_az.alt.deg, self.camera_alt_az.az.deg, now, TelescopeAltAzFrame
            )
            await self.camera_alignment_handler.add_alignment_position(
                altaz=sky_alt_az, telescope=camera_alt_az
            )
            self.log.debug(
                "New alignment point SkyAltAz=%s and CameraAltAz=%s.",
                sky_alt_az.to_string("dms"),
                camera_alt_az.to_string("dms"),
            )
        if self.controller_type in [MotorControllerType.MOTORS_ONLY, MotorControllerType.CAMERA_AND_MOTORS]:
            assert self.motor_controller_alt is not None
            assert self.motor_controller_az is not None
            self.motor_controller_alt.position = sky_alt_az.alt
            self.motor_controller_az.position = sky_alt_az.az

    async def set_slew_rate(self, cmd: str) -> None:
        """Set the slew rate.

        The command is part of the LX200 protocol.

        Parameters
        ----------
        cmd : `str`
            A set slew rate command.
        """
        await asyncio.sleep(MILLISECOND)

        match cmd:
            case "RC":
                self.slew_rate = SlewRate.CENTERING
            case "RG":
                self.slew_rate = SlewRate.GUIDING
            case "RM":
                self.slew_rate = SlewRate.FIND
            case "RS":
                self.slew_rate = SlewRate.HIGH
            case _:
                raise ValueError(f"Received unknown slew rate command {cmd}.")

    async def slew_to(self, ra_str: str, dec_str: str) -> str:
        """Instruct the mount to slew to the target RA and DEC if possible.

        Parameters
        ----------
        ra_str: `str`
            The Right Ascension of the mount in degrees. The format is
            "HH:mm:ss".
        dec_str: `str`
            The Declination of the mount in degrees. The format is "+dd*mm:ss".

        Returns
        -------
        slew_possible: 0 or 1
            0 means in reach, 1 not.
        """
        assert self.motor_controller_alt is not None
        assert self.motor_controller_az is not None

        now = DatetimeUtil.get_timestamp()
        self.target_radec = await get_skycoord_from_ra_dec_str(ra_str=ra_str, dec_str=dec_str)
        self.log.debug("slew_to Set target_radec to %s.", self.target_radec.to_string("hmsdms"))
        motor_alt_az = await get_altaz_from_radec(ra_dec=self.target_radec, timestamp=now)

        # Compute slew times.
        az_slew_time = await self.motor_controller_az.estimate_slew_time(motor_alt_az.az)
        alt_slew_time = await self.motor_controller_alt.estimate_slew_time(motor_alt_az.alt)

        slew_time = max(az_slew_time, alt_slew_time)

        # Compute AltAz at the end of the slew.
        fut_time = motor_alt_az.obstime + slew_time * u.second
        fut_altaz_frame = AltAz(obstime=fut_time, location=get_observing_location())
        mount_alt_az_after_slew = self.target_radec.transform_to(fut_altaz_frame)

        self.slew_direction = SlewDirection.NONE
        if mount_alt_az_after_slew.alt.value > 0:
            self.slew_rate = SlewRate.HIGH
            await self.motor_controller_az.move(mount_alt_az_after_slew.az)
            await self.motor_controller_alt.move(mount_alt_az_after_slew.alt)
            return "0"
        else:
            return "1"

    async def slew_in_direction(self, cmd: str) -> None:
        """Slew the mount in the provided direction.

        Parameters
        ----------
        cmd : `str`
            The command that specifies which direction to slew to.
        """
        assert self.motor_controller_alt is not None
        assert self.motor_controller_az is not None

        match cmd:
            case "Mn":
                self.slew_direction = SlewDirection.UP
                await self.motor_controller_alt.move(NINETY, self.slew_rate)
            case "Me":
                self.slew_direction = SlewDirection.LEFT
                await self.motor_controller_az.move(
                    self.motor_controller_az.position - NINETY, self.slew_rate
                )
            case "Ms":
                self.slew_direction = SlewDirection.DOWN
                await self.motor_controller_alt.move(ZERO, self.slew_rate)
            case "Mw":
                self.slew_direction = SlewDirection.RIGHT
                await self.motor_controller_az.move(
                    self.motor_controller_az.position + NINETY, self.slew_rate
                )
            case _:
                self.slew_direction = SlewDirection.NONE
                raise ValueError(f"Received unknown slew direction command {cmd}.")
        self.log.debug(f"SlewDirection = {self.slew_direction.name}")

    async def stop_slew(self) -> None:
        """Stop the slew of both motors."""
        assert self.motor_controller_alt is not None
        assert self.motor_controller_az is not None

        self.slew_direction = SlewDirection.NONE
        await self.motor_controller_az.stop_motion()
        await self.motor_controller_alt.stop_motion()
        target_altaz = await get_skycoord_from_alt_az(
            alt=self.motor_controller_alt.target_position.deg,
            az=self.motor_controller_az.target_position.deg,
            timestamp=DatetimeUtil.get_timestamp(),
            frame=TelescopeAltAzFrame,
        )
        self.target_radec = await get_radec_from_altaz(target_altaz)
        self.log.debug("stop_slew Set target_radec to %s.", self.target_radec.to_string("hmsdms"))

    async def location_updated(self) -> None:
        """Update the location.

        Also stay pointed at the same altitude and azimuth.
        """
        pass

    async def __aenter__(self) -> MountController:
        await self.start()
        return self

    async def __aexit__(
        self,
        _type: None | BaseException,
        _value: None | BaseException,
        _traceback: None | types.TracebackType,
    ) -> None:
        await self.stop()
