__all__ = ["EmulatedMotorController"]

import asyncio
import logging
import typing

from astropy.coordinates import Angle

from ..datetime_util import DatetimeUtil
from ..motor.base_motor_controller import BaseMotorController
from ..motor.trajectory import Trajectory, TrajectorySegment, accelerated_pos_and_vel

# Time to wait for the stepper motor to report that it is attached [ms].
ATTACH_WAIT_TIME = 2000
# The maximum velocity of the stepper motor [steps/s].
MAX_VELOCITY = 100000.0
# The maximum acceleration of the stepper motor [steps/s^2].
MAX_ACCEL = 50000.0


class EmulatedStepper:
    """Emulate a Phidgets stepper motor."""

    def __init__(self) -> None:
        self.log = logging.getLogger(type(self).__name__)

        # Keep track of the current time, position and velocity.
        self._time = 0.0
        self._command_time = 0.0
        self._position = 0.0
        self._velocity = 0.0
        # Absolute value of the acceleration to use. The sign is not checked.
        self._acceleration = 0.0
        self._start_position = 0.0
        self._start_velocity = 0.0
        # The actual acceleration used for the computation of position and velocity. The sign is set based on
        # the position difference or the sign of the velocity.
        self._a = 0.0

        self._data_interval = 0.0
        self._engaged = False
        self._hub_port = 0

        self._trajectory: Trajectory | None = None

        self._target_position = 0.0
        self._velocity_limit = 0.0

        # Callback handlers.
        self._on_attach_handler: typing.Callable | None = None
        self._on_detach_handler: typing.Callable | None = None
        self._on_error_handler: typing.Callable | None = None
        self._on_position_change_handler: typing.Callable | None = None
        self._on_velocity_change_handler: typing.Callable | None = None

        # Position loop task, which gets initialized to ``done``.
        self._position_loop_task: asyncio.Future = asyncio.Future()
        self._position_loop_task.set_result(None)

    async def _position_loop(self) -> None:
        """Execute the position loop.

        Compute the position and velocity for the current time. Call the position and velocity callbacks if
        the position respectively the velocity have changed. The loop delay is non-drifiting.
        """
        start_time = DatetimeUtil.get_timestamp()
        while True:
            # Keep track of old time, position and velocity for change handlers.
            old_position = self._position
            old_velocity = self._velocity

            self._compute_position_and_velocity()

            # Call change handlers if necessary.
            if old_position != self._position and self._on_position_change_handler:
                self._on_position_change_handler(self, self._position)
            if old_velocity != self._velocity and self._on_velocity_change_handler:
                self._on_velocity_change_handler(self, self._velocity)

            # Compute the remainder of the wait time to avoid drift.
            remainder = (
                DatetimeUtil.get_timestamp() - start_time
            ) % self._data_interval
            # Now sleep the remainder of the wait time.
            await asyncio.sleep(self._data_interval - remainder)

    def _compute_position_and_velocity(self) -> None:
        """Use the computed trajectory to compute the position and velocity for the current time."""
        if self._trajectory is not None and len(self._trajectory.segments) != 0:
            now = DatetimeUtil.get_timestamp()
            time_since_command_time = now - self._command_time

            segment_to_use: TrajectorySegment | None = None
            for segment in reversed(self._trajectory.segments):
                if time_since_command_time >= segment.start_time:
                    segment_to_use = segment
                    break

            assert (
                segment_to_use is not None
            ), f"{segment_to_use=}, {self._trajectory.segments=}"
            time_to_use = time_since_command_time - segment_to_use.start_time
            self._position, self._velocity = accelerated_pos_and_vel(
                start_position=segment_to_use.start_position,
                start_velocity=segment_to_use.start_velocity,
                acceleration=segment_to_use.acceleration,
                time=time_to_use,
            )

    def close(self) -> None:
        """Emulate the Phidgets close method."""
        self.log.debug("Setting is_open = False")
        if self._on_detach_handler:
            self._on_detach_handler(self)

    def get_min_data_interval(self) -> float:
        """Emulate the Phidgets getMinDataInterval method."""
        return 0.1

    def open_wait_for_attachment(self, timeout: float) -> None:
        """Emulate the Phidgets openWaitForAttachment method."""
        self.log.debug(f"Setting is_open = True and ignoring {timeout=}")
        if self._on_attach_handler:
            self._on_attach_handler(self)

    def set_acceleration(self, acceleration: float) -> None:
        """Emulate the Phidgets setAcceleration method."""
        self._acceleration = acceleration
        self._trajectory = Trajectory(acceleration)

    def set_data_interval(self, data_interval: float) -> None:
        """Emulate the Phidgets setDataInterval method."""
        self._data_interval = data_interval

    def set_engaged(self, engaged: bool) -> None:
        """Emulate the Phidgets setEngaged method."""
        self._engaged = engaged
        if engaged:
            self._position_loop_task = asyncio.create_task(self._position_loop())
        else:
            if not self._position_loop_task.done():
                try:
                    self._position_loop_task.cancel()
                except Exception:
                    if self._on_error_handler:
                        self._on_error_handler(
                            code=1, description="Position loop task got cancelled."
                        )

    def set_hub_port(self, hub_port: int) -> None:
        """Emulate the Phidgets setHubPort method."""
        self._hub_port = hub_port

    def set_on_attach_handler(self, handler: typing.Callable) -> None:
        """Emulate the Phidgets setOnAttachHandler method."""
        self._on_attach_handler = handler

    def set_on_detach_handler(self, handler: typing.Callable) -> None:
        """Emulate the Phidgets setOnDetachHandler method."""
        self._on_detach_handler = handler

    def set_on_position_change_handler(self, handler: typing.Callable) -> None:
        """Emulate the Phidgets setOnPositionChangeHandler method."""
        self._on_position_change_handler = handler

    def set_on_velocity_change_handler(self, handler: typing.Callable) -> None:
        """Emulate the Phidgets setOnVelocityChangeHandler method."""
        self._on_velocity_change_handler = handler

    def set_target_position(self, target_position: float) -> None:
        """Emulate the Phidgets setTargetPosition method.

        Apart from setting the target position, also compute the paramters for the motion.

        Calling this method will make the motor start moving from the current position if the target position
        is not equal to the current position or if the current velocity is not 0.

        Parameters
        ----------
        target_position : `float`
            The position to move to.
        """
        self._target_position = target_position
        self._start_position = self._position
        self._start_velocity = self._velocity
        self._command_time = DatetimeUtil.get_timestamp()

        if self._trajectory is not None:
            self._trajectory.set_target_position_and_velocity(
                curr_pos=self._start_position,
                curr_vel=self._start_velocity,
                target_position=target_position,
                max_velocity=self._velocity_limit,
            )

    def set_velocity_limit(self, velocity_limit: float) -> None:
        """Emulate the Phidgets setVelocityLimit method."""
        self._velocity_limit = velocity_limit


class EmulatedMotorController(BaseMotorController):
    """Emulate a motor controller.

    See `BaseMotorController`.
    """

    def __init__(
        self,
        initial_position: Angle,
        log: logging.Logger,
        conversion_factor: Angle,
        hub_port: int,
    ) -> None:
        name = BaseMotorController.ALT if hub_port == 0 else BaseMotorController.AZ
        super().__init__(log=log, name=name, conversion_factor=conversion_factor)
        self._max_velocity = MAX_VELOCITY
        self._max_acceleration = MAX_ACCEL

        self.stepper = EmulatedStepper()
        self.stepper.set_hub_port(hub_port)
        self.stepper.set_on_attach_handler(self.on_attach)
        self.stepper.set_on_detach_handler(self.on_detach)
        self.stepper.set_on_position_change_handler(self.on_position_change)
        self.stepper.set_on_velocity_change_handler(self.on_velocity_change)

        self._position_offset = round(
            (initial_position / self._conversion_factor).value
        )

    async def connect(self) -> None:
        """Connect the stepper motor."""
        assert self.stepper is not None
        try:
            self.stepper.open_wait_for_attachment(ATTACH_WAIT_TIME)
        except Exception as e:
            raise RuntimeError(e)
        assert self.attached
        self.stepper.set_engaged(True)
        self.stepper.set_acceleration(self._max_acceleration)
        self.stepper.set_data_interval(self.stepper.get_min_data_interval())

    async def disconnect(self) -> None:
        """Disconnect the stepper motor."""
        assert self.stepper is not None
        self.stepper.set_engaged(False)
        self.stepper.close()
        assert not self.attached

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
        assert self.stepper is not None
        self.stepper.set_velocity_limit(abs(max_velocity_in_steps))
        self.stepper.set_target_position(target_position_in_steps)
