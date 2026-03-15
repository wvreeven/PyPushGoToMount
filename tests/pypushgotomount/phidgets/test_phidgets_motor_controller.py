import logging
import typing
from unittest import IsolatedAsyncioTestCase, mock

import astropy.units as u
import pypushgotomount
from astropy.coordinates import Angle


class TestPhidgetsMotorController(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.log = logging.getLogger(type(self).__name__)
        self.pmc = pypushgotomount.phidgets.PhidgetsMotorController(
            initial_position=Angle(0.0 * u.deg),
            log=self.log,
            hub_port=0,
            is_remote=False,
            conversion_factor=Angle(0.0001 * u.deg),
        )
        mock_stepper = mock.MagicMock()
        mock_stepper.openWaitForAttachment.side_effect = self.attach
        mock_stepper.close.side_effect = self.detach
        self.pmc.stepper = mock_stepper

    async def test_move(self) -> None:
        await self.pmc.connect()
        assert self.pmc.attached
        await self.pmc.move_to_target_position_at_velocity(
            target_position_in_steps=10000.0, max_velocity_in_steps=10000.0
        )
        await self.pmc.track_at_velocity(max_velocity_in_steps=100.0)
        await self.pmc.disconnect()
        assert not self.pmc.attached

    def attach(self, _: typing.Any) -> None:
        self.pmc.attached = True

    def detach(self) -> None:
        self.pmc.attached = False
