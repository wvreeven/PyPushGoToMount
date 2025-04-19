__all__ = ["PhidgetsMotorController"]

import logging

from astropy.coordinates import Angle
from Phidget22.Devices.Stepper import Stepper, StepperControlMode
from Phidget22.Net import Net, PhidgetServerType
from Phidget22.PhidgetException import PhidgetException

from ..motor.base_motor_controller import BaseMotorController

# The maximum acceleration of the stepper motor [deg/sec].
ACCELERATION = 60000
# Time to wait for the stepper motor to report that it is attached.
ATTACH_WAIT_TIME = 2000


class PhidgetsMotorController(BaseMotorController):
    """Phidgets motor controller.

    See `BaseMotorController`.
    """

    def __init__(
        self,
        initial_position: Angle,
        log: logging.Logger,
        conversion_factor: Angle,
        hub_port: int,
        is_remote: bool = True,
    ) -> None:
        name = "Alt" if hub_port == 0 else "Az"
        super().__init__(log=log, name=name, conversion_factor=conversion_factor)

        if is_remote:
            Net.enableServerDiscovery(PhidgetServerType.PHIDGETSERVER_DEVICEREMOTE)

        self._max_velocity = ACCELERATION
        self._max_acceleration = ACCELERATION

        self.stepper = Stepper()
        self.stepper.setHubPort(hub_port)
        self.stepper.setIsRemote(is_remote)
        self.stepper.setOnAttachHandler(self.on_attach)
        self.stepper.setOnDetachHandler(self.on_detach)
        self.stepper.setOnPositionChangeHandler(self.on_position_change)
        self.stepper.setOnVelocityChangeHandler(self.on_velocity_change)
        self.stepper.setOnErrorHandler(self.on_error)

        self._position_offset = round(
            (initial_position / self._conversion_factor).value
        )

    def on_error(self, code: int, description: str) -> None:
        self.log.error(f"{code=!s} -> {description=!s}")

    async def connect(self) -> None:
        """Connect the stepper motor."""
        try:
            self.stepper.openWaitForAttachment(ATTACH_WAIT_TIME)
        except PhidgetException as e:  # type: ignore  # noqa
            raise RuntimeError(e)
        assert self.attached
        self.stepper.setEngaged(True)
        self.stepper.setAcceleration(ACCELERATION)
        self.stepper.setDataInterval(self.stepper.getMinDataInterval())

    async def disconnect(self) -> None:
        """Disconnect the stepper motor."""
        self.stepper.setEngaged(False)
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
        # Switch control mode if necessary.
        if self.stepper.getControlMode() != StepperControlMode.CONTROL_MODE_STEP:
            self.stepper.setEngaged(False)
            self.stepper.setControlMode(StepperControlMode.CONTROL_MODE_STEP)
            self.stepper.setEngaged(True)

        self.stepper.setVelocityLimit(abs(max_velocity_in_steps))
        self.stepper.setTargetPosition(target_position_in_steps)
