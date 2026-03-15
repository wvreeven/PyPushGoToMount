import math
from unittest import IsolatedAsyncioTestCase

import numpy
import pypushgotomount
import pytest


class TestTrajectory(IsolatedAsyncioTestCase):
    async def test_pos(self) -> None:
        self.trajectory = pypushgotomount.motor.Trajectory(max_acceleration=50000.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=0.0, target_position=0.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=100000.0, target_position=0.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=-100000.0, target_position=0.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=0.0, target_position=1000000.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=0.0, target_position=100000.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=0.0, target_position=200000.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=0.0, target_position=-200000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=0.0, target_position=1000000.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=50000.0, target_position=1000000.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=-50000.0, target_position=1000000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=50000.0, target_position=1000000.0)
        self.assert_trajectory(curr_pos=-90000.0, curr_vel=-50000.0, target_position=-1000000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=-50000.0, target_position=1000000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=50000.0, target_position=-1000000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=50000.0, target_position=10000.0)
        self.assert_trajectory(curr_pos=-90000.0, curr_vel=-50000.0, target_position=-10000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=50000.0, target_position=-10000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=-50000.0, target_position=100000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=50000.0, target_position=-100000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=-50000.0, target_position=10000.0)
        self.assert_trajectory(curr_pos=90000.0, curr_vel=50000.0, target_position=100000.0)
        self.assert_trajectory(curr_pos=-90000.0, curr_vel=-50000.0, target_position=-100000.0)
        self.assert_trajectory(curr_pos=10000.0, curr_vel=0.0, target_position=10000.0)
        self.assert_trajectory(curr_pos=10000.0, curr_vel=100000.0, target_position=10000.0)
        self.assert_trajectory(curr_pos=10000.0, curr_vel=-100000.0, target_position=10000.0)

        # No target position.
        self.assert_trajectory(curr_pos=0.0, curr_vel=0.0, target_position=None, max_velocity=50.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=25.0, target_position=None, max_velocity=50.0)
        self.assert_trajectory(curr_pos=0.0, curr_vel=50.0, target_position=None, max_velocity=50.0)

    def assert_trajectory(
        self,
        curr_pos: float,
        curr_vel: float,
        target_position: float | None,
        max_velocity: float = 100000.0,
    ) -> None:
        self.trajectory.set_target_position_and_velocity(
            curr_pos=curr_pos,
            curr_vel=curr_vel,
            target_position=target_position,
            max_velocity=max_velocity,
        )
        if target_position is None:
            if math.isclose(curr_vel, max_velocity):
                assert len(self.trajectory.segments) == 1
                assert math.isclose(self.trajectory.segments[0].start_velocity, max_velocity)
                assert math.isclose(self.trajectory.segments[0].acceleration, 0.0)
            else:
                assert len(self.trajectory.segments) == 2
                assert math.isclose(self.trajectory.segments[1].start_velocity, max_velocity)
                assert math.isclose(self.trajectory.segments[1].acceleration, 0.0)
        elif curr_pos == target_position and math.isclose(curr_vel, 0.0):
            assert len(self.trajectory.segments) == 1
            assert self.trajectory.segments[0].start_position == curr_pos
            assert self.trajectory.segments[0].start_velocity == curr_vel
            assert math.isclose(self.trajectory.segments[0].acceleration, 0.0)
        else:
            assert len(self.trajectory.segments) > 1
            for i in range(1, len(self.trajectory.segments)):
                segment0 = self.trajectory.segments[i - 1]
                segment1 = self.trajectory.segments[i]
                assert not numpy.iscomplexobj(segment0.start_position)
                assert not numpy.iscomplexobj(segment0.start_velocity)
                assert not numpy.iscomplexobj(segment0.acceleration)
                assert not numpy.iscomplexobj(segment0.start_time)
                assert not numpy.iscomplexobj(segment1.start_position)
                assert not numpy.iscomplexobj(segment1.start_velocity)
                assert not numpy.iscomplexobj(segment1.acceleration)
                assert not numpy.iscomplexobj(segment1.start_time)
                pos0, vel0 = pypushgotomount.motor.accelerated_pos_and_vel(
                    segment0.start_position,
                    segment0.start_velocity,
                    segment0.acceleration,
                    segment1.start_time - segment0.start_time,
                )
                pos1, vel1 = pypushgotomount.motor.accelerated_pos_and_vel(
                    segment1.start_position,
                    segment1.start_velocity,
                    segment1.acceleration,
                    0.0,
                )
                assert pos0 == pytest.approx(pos1, abs=1e-9)
                assert vel0 == pytest.approx(vel1, abs=1e-9)

            last_segment = self.trajectory.segments[-1]
            assert last_segment.start_position == target_position
            assert math.isclose(last_segment.start_velocity, 0.0)
            assert math.isclose(last_segment.acceleration, 0.0)
