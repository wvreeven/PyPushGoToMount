from __future__ import annotations

__all__ = ["BaseMotorController"]

import asyncio
import logging
import types
import typing
from abc import ABC, abstractmethod

import astropy.units as u
from astropy.coordinates import Angle

from ..enums import MILLISECOND, MotorControllerState, SlewRate
from .trajectory import Trajectory, accelerated_pos_and_vel

# An angle of 180º.
ONE_EIGHTY = Angle(180.0 * u.deg)
# An angle of 360º.
THREE_SIXTY = Angle(360.0 * u.deg)
# Wrap angle for altitude.
ALT_WRAP = ONE_EIGHTY
# Wrap angle for azimuth.
AZ_WRAP = THREE_SIXTY


class BaseMotorController(ABC):
    """Base class for all motor controllers.

    This class takes care of as much logic for controlling a stepper motor in a generic way.

    The base class is responsible for:

      * Callbacks for when the physical motor attaches and detaches.
      * Callbacks for when the position or velocity changes.
      * Converting the number of motor steps to an angle and back for position, velocity and acceleration,
        taking the gear reduction and position offset into account.
      * Move to the given target position with the specified maximum velocity.
      * Track to the given target position for the given duration.
      * Stop the current motion.
      * Async context manager methods.

    Implementing classes need to implement the following methods:

      * Connect and disconnect the physical motor.
      * Set the target position in steps and the maximum velocity in steps/sec in the physical motors.
    """

    # Name for altitude motors.
    ALT = "Alt"
    # Name for azimuth motors.
    AZ = "Az"

    def __init__(self, log: logging.Logger, name: str, conversion_factor: Angle) -> None:
        self.name = name
        self.log = log.getChild(f"{type(self).__name__} {self.name}")

        self._position = 0.0
        self._velocity = 0.0
        self._max_velocity = 0.0
        self._max_acceleration = 0.0
        self._conversion_factor = conversion_factor
        self.log.info(
            f'Conversion factor set to {conversion_factor.deg}º == {conversion_factor.deg * 3600.0:.4f}".'
        )
        self._position_offset = 0.0
        self.state = MotorControllerState.STOPPED

        # Target position.
        self.target_position = Angle(0.0 * u.deg)

        # Some necessary constants that need to be computed once the conversion factor is known.
        self._one_eighty_steps = (ONE_EIGHTY / self._conversion_factor).value
        self._three_sixty_steps = (THREE_SIXTY / self._conversion_factor).value

        self.attached = False

    @property
    def position(self) -> Angle:
        """The motor position [steps] as an astropy `Angle`."""
        pos = (self._position + self._position_offset) * self._conversion_factor
        return pos.wrap_at(ALT_WRAP if self.name == BaseMotorController.ALT else AZ_WRAP)

    @position.setter
    def position(self, position: Angle) -> None:
        """Setter for the motor position [astropy `Angle`] which gets converted to steps."""
        position_value = (position / self._conversion_factor).value
        self._position_offset = position_value - self._position

    @property
    def velocity(self) -> Angle:
        """The motor velocity [steps/sec] as an astropy `Angle` per sec."""
        return self._velocity * self._conversion_factor

    @property
    def max_velocity(self) -> Angle:
        """The motor maximum velocity [steps/sec] as an astropy `Angle` per sec."""
        return self._max_velocity * self._conversion_factor

    @property
    def max_acceleration(self) -> Angle:
        """The motor maximum acceleration [steps/sec^2] as an astropy `Angle` per sec^2."""
        return self._max_acceleration * self._conversion_factor

    async def _get_target_position_in_steps(self, target_position: Angle) -> int:
        """Get the target position in steps.

        This takes both the conversion factor and the position offset into account. The number of steps is
        increased or decreased to match the multiple of steps equivalent to 180 degrees of the motor position.

        Parameters
        ----------
        target_position : `Angle`
            The target position.

        Returns
        -------
        int
            The target position [steps].
        """
        await asyncio.sleep(MILLISECOND)

        diff = (target_position - self.position).wrap_at(ONE_EIGHTY)
        diff_in_steps = (diff / self._conversion_factor).value
        target_position_in_steps = self._position + diff_in_steps
        return target_position_in_steps

    def on_attach(self, _: typing.Any) -> None:
        """On attach callback."""
        self.log.info("Attach stepper!")
        self.attached = True

    def on_detach(self, _: typing.Any) -> None:
        """On detach callback."""
        self.log.info("Detach stepper!")
        self.attached = False

    def on_position_change(self, _: typing.Any, current_position: int) -> None:
        """On position change callback.

        Parameters
        ----------
        _: `typing.Any`
            An instance of the stepper class.
        current_position: `int`
            The current position of the stepper motor [steps].
        """
        self._position = current_position

    def on_velocity_change(self, _: typing.Any, current_velocity: int) -> None:
        """On velocity change callback.

        Parameters
        ----------
        _: `typing.Any`
            An instance of the stepper class.
        current_velocity: `int`
            The current velocity of the stepper motor [steps/sec].
        """
        self._velocity = current_velocity

    async def stop_motion(self) -> None:
        """Stop the current motion.

        If currently moving, this will slow down the motion at the maximum acceleration until stopped. In
        order to do this, the time to stop from the current velocity assuming the maximum acceleration is
        computed. Based on that time, the resulting target position is computed. The target position and
        maximum velocity then are passed on to the method that sets it in the physical motor.

        If already stopped then this will have no effect.
        """
        self.state = MotorControllerState.STOPPING
        position_to_start_stopping = self._position
        max_velocity_in_steps = self._velocity
        if max_velocity_in_steps > 0:
            accel = -self._max_acceleration
        elif max_velocity_in_steps < 0:
            accel = self._max_acceleration
        else:
            self.log.exception(f"{max_velocity_in_steps=}. Ignoring.")
            return
        time_needed_to_stop = abs(max_velocity_in_steps / accel)
        target_position_in_steps, _ = accelerated_pos_and_vel(
            position_to_start_stopping,
            max_velocity_in_steps,
            accel,
            time_needed_to_stop,
        )
        self.target_position = target_position_in_steps * self._conversion_factor
        target_position_in_steps = await self._get_target_position_in_steps(self.target_position)

        await self.move_to_target_position_at_velocity(target_position_in_steps, max_velocity_in_steps)

    async def move(self, target_position: Angle, slew_rate: SlewRate = SlewRate.HIGH) -> None:
        """Move to the indicated target position at the maximum velocity.

        Parameters
        ----------
        target_position: `Angle`
            The target position to slew to [º].
        slew_rate: `SlewRate`
            The slew rate to apply. This determines the maximnum speed at which a slew is performed. Defaults
            to HIGH, which is the highest rate.
        """
        self.state = MotorControllerState.SLEWING
        self.target_position = target_position
        target_position_in_steps = await self._get_target_position_in_steps(target_position)
        max_velocity_in_steps = self._max_velocity * slew_rate / SlewRate.HIGH

        await self.move_to_target_position_at_velocity(target_position_in_steps, max_velocity_in_steps)

    async def track(self, target_position: Angle, timediff: float) -> None:
        """Track the provided target position over the provided time difference.

        The velocity to track with is computed based on the difference in steps, computed from the current
        position and the target position, and the time difference.

        Parameters
        ----------
        target_position : `Angle`
            The target position to track to.
        timediff : `float`
            The amount of time to take to track to the target position.
        """
        target_position_in_steps = (
            await self._get_target_position_in_steps(target_position) + self._position_offset
        )
        position_in_steps = self._position + self._position_offset
        max_velocity_in_steps = (target_position_in_steps - position_in_steps) / timediff
        await self.track_at_velocity(max_velocity_in_steps=max_velocity_in_steps)

    async def estimate_slew_time(self, target_position: Angle) -> float:
        """Estimate the slew time to the target position.

        Parameters
        ----------
        target_position : `Angle`
            The target position to estimate the slew time for.

        Returns
        -------
        float
            The estimated slew time to the target position.
        """
        await asyncio.sleep(MILLISECOND)
        trajectory = Trajectory(max_acceleration=self._max_acceleration)
        target_position_in_steps = await self._get_target_position_in_steps(target_position)
        position_in_steps = self._position - self._position_offset
        trajectory.set_target_position_and_velocity(
            curr_pos=position_in_steps,
            curr_vel=self._velocity,
            target_position=target_position_in_steps,
            max_velocity=self._max_velocity,
        )
        return trajectory.segments[-1].start_time

    @abstractmethod
    async def connect(self) -> None:
        """Connect the stepper motor."""
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect the stepper motor."""
        raise NotImplementedError

    @abstractmethod
    async def move_to_target_position_at_velocity(
        self, target_position_in_steps: float, max_velocity_in_steps: float
    ) -> None:
        """Move the motor to the provided position at the provided maximum velocity and stop.

        Parameters
        ----------
        target_position_in_steps : `float`
            The target position [steps].
        max_velocity_in_steps : `float`
            The maximum velocity [steps/sec].
        """
        raise NotImplementedError

    @abstractmethod
    async def track_at_velocity(self, max_velocity_in_steps: float) -> None:
        """Track the motor at the provided maximum velocity and stop.

        Parameters
        ----------
        max_velocity_in_steps : `float`
            The maximum velocity [steps/sec].
        """
        raise NotImplementedError

    async def __aenter__(self) -> BaseMotorController:
        await self.connect()
        return self

    async def __aexit__(
        self,
        _type: None | BaseException,
        _value: None | BaseException,
        _traceback: None | types.TracebackType,
    ) -> None:
        await self.disconnect()
