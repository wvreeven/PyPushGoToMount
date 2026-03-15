__all__ = ["Trajectory", "TrajectorySegment", "accelerated_pos_and_vel"]

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectorySegment:
    """Segment of a trajectory.

    Attributes
    ----------
    start_time : `float`
        The start time.
    start_position : `float`
        The start position.
    start_velocity : `float`
        The start velocity.
    acceleration : `float`
        The acceleration.
    """

    start_time: float
    start_position: float
    start_velocity: float
    acceleration: float


def accelerated_pos_and_vel(
    start_position: float, start_velocity: float, acceleration: float, time: float
) -> tuple[float, float]:
    """Compute the position and velocity for accelerated motion defined by the provided parameters at the
    provided time.

    Parameters
    ----------
    start_position : `float`
        The position [any unit] at t = 0s.
    start_velocity : `float`
        The velocity [any unit/sec] at t = 0s.
    acceleration : `float`
        The acceleration [any unit/sec^2].
    time : `float`
        The time [s] for which the position and velocity are calculated.

    Returns
    -------
    p, v : `tuple`[`float`]
        The position and velocity at the provided time.

    Notes
    -----
    All positions, velocities and accelerations are unitless to make this a generic trajectory that can be
    applied to degrees, radians, or motor steps (or even other units if desirable). It is assumed that all
    values have the same base unit, e.g. deg for positions in deg, velocities in deg/sec and accelerations in
    deg/sec^2.
    """
    velocity = start_velocity + acceleration * time
    position = start_position + (velocity + start_velocity) * time / 2.0
    return position, velocity


class Trajectory:
    """Class representing the trajectory.

    The trajectory starts at the current position with the current velocity and acceleration and ends at the
    target position with both velocity and acceleration 0.0.

    The start time of the first trajectory segment is always 0.0 sec.

    Notes
    -----
    All positions, velocities and accelerations are unitless to make this a generic trajectory that can be
    applied to degrees, radians, or motor steps (or even other units if desirable). It is assumed that all
    values have the same base unit, e.g. deg for positions in deg, velocities in deg/sec and accelerations in
    deg/sec^2.
    """

    def __init__(self, max_acceleration: float) -> None:
        self.segments: list[TrajectorySegment] = []
        self._max_acceleration = max_acceleration

    def __repr__(self) -> str:
        return f"Trajectory[{list(self.segments)}]"

    def set_target_position_and_velocity(
        self,
        curr_pos: float,
        curr_vel: float,
        target_position: float | None,
        max_velocity: float,
    ) -> None:
        """Set the target position and maximum velocity.

        Using the current position and velocity, all segements for the entire trajectory are computed.

        Parameters
        ----------
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        target_position : `float` | `None`
            The target position [any unit]. If None then motion will continue forever at max velocity.
        max_velocity : `float`
            The maximum velocity [any unit/sec].
        """
        self.segments = []
        if target_position is None:
            self.handle_no_target_position(curr_pos, curr_vel, max_velocity)
        elif target_position == curr_pos:
            self.handle_target_same_as_pos(curr_pos, curr_vel, target_position, max_velocity)
        else:
            self.handle_target_not_same_as_pos(curr_pos, curr_vel, target_position, max_velocity)

        self.consolidate_segments()

    def handle_no_target_position(
        self,
        curr_pos: float,
        curr_vel: float,
        max_velocity: float,
    ) -> None:
        """Compute the trajectory for the case where there is no target position.

        Parameters
        ----------
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        max_velocity : `float`
            The maximum velocity [any unit/sec].
        """
        if math.isclose(curr_vel, max_velocity):
            # Moving at the current velocity, so done.
            self.segments = [
                TrajectorySegment(
                    start_time=0.0,
                    start_position=curr_pos,
                    start_velocity=curr_vel,
                    acceleration=0.0,
                )
            ]
        else:
            delta_v = max_velocity - curr_vel
            sign_vel = math.copysign(1, delta_v)
            accel = self._max_acceleration * sign_vel
            time_to_max_vel = delta_v / accel
            pos_at_max_vel, vel_at_max_vel = accelerated_pos_and_vel(
                curr_pos, curr_vel, accel, time_to_max_vel
            )
            self.segments = [
                TrajectorySegment(
                    start_time=0.0,
                    start_position=curr_pos,
                    start_velocity=curr_vel,
                    acceleration=accel,
                ),
                TrajectorySegment(
                    start_time=time_to_max_vel,
                    start_position=pos_at_max_vel,
                    start_velocity=vel_at_max_vel,
                    acceleration=0.0,
                ),
            ]

    def handle_target_same_as_pos(
        self,
        curr_pos: float,
        curr_vel: float,
        target_position: float,
        max_velocity: float,
    ) -> None:
        """Compute the trajectory for the case where the target position is the same as the start position.

        Parameters
        ----------
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        target_position : `float`
            The target position [any unit].
        max_velocity : `float`
            The maximum velocity [any unit/sec].
        """
        if math.isclose(curr_vel, 0.0):
            # At the current position and not moving, so done.
            self.segments = [
                TrajectorySegment(
                    start_time=0.0,
                    start_position=curr_pos,
                    start_velocity=curr_vel,
                    acceleration=0.0,
                )
            ]
        else:
            self._determine_all_trajectory_segments(curr_pos, curr_vel, target_position, max_velocity)

    def handle_target_not_same_as_pos(
        self,
        curr_pos: float,
        curr_vel: float,
        target_position: float,
        max_velocity: float,
    ) -> None:
        """Compute the trajectory for the case where the target position is not the same as the start
        position.

        Parameters
        ----------
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        target_position : `float`
            The target position [any unit].
        max_velocity : `float`
            The maximum velocity [any unit/sec].
        """
        sign_pos = math.copysign(1, target_position - curr_pos)
        sign_vel = math.copysign(1, curr_vel)
        if sign_pos == sign_vel or math.isclose(curr_vel, 0.0):
            if target_position > curr_pos:
                max_vel = max_velocity
                accel = self._max_acceleration
            else:
                max_vel = -max_velocity
                accel = -self._max_acceleration
            self.segments = self._determine_trajectory_segments(
                start_time=0.0,
                curr_pos=curr_pos,
                curr_vel=curr_vel,
                target_position=target_position,
                max_vel=max_vel,
                accel=accel,
            )
        else:
            self._determine_all_trajectory_segments(curr_pos, curr_vel, target_position, max_velocity)

    def _determine_all_trajectory_segments(
        self,
        curr_pos: float,
        curr_vel: float,
        target_position: float,
        max_velocity: float,
    ) -> None:
        time_to_stop = abs(curr_vel / self._max_acceleration)
        if curr_vel > 0.0:
            max_vel = -max_velocity
            accel = -self._max_acceleration
        else:
            max_vel = max_velocity
            accel = self._max_acceleration
        pos_when_stopped, vel_when_stopped = accelerated_pos_and_vel(curr_pos, curr_vel, accel, time_to_stop)
        self.segments = [
            TrajectorySegment(
                start_time=0.0,
                start_position=curr_pos,
                start_velocity=curr_vel,
                acceleration=accel,
            )
        ] + self._determine_trajectory_segments(
            start_time=time_to_stop,
            curr_pos=pos_when_stopped,
            curr_vel=vel_when_stopped,
            target_position=target_position,
            max_vel=max_vel,
            accel=accel,
        )

    def _determine_trajectory_segments(
        self,
        start_time: float,
        curr_pos: float,
        curr_vel: float,
        target_position: float,
        max_vel: float,
        accel: float,
    ) -> list[TrajectorySegment]:
        """Determine the trajectory segments.

        First the segments are determined assuming that the maximum velocity can be reached. If then it turns
        out that that isn't the case, the segments are determined again without reaching the maximum velocity.

        Parameters
        ----------
        start_time : `float`
            The start time [sec].
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        target_position : `float`
            The target position [any unit].
        max_vel : `float`
            The maximum velocity [any unit/sec].
        accel : `float`
            The acceleration [any unit/sec^2].

        Returns
        -------
        `list`[`TrajectorySegment`]
            The segments that form the entire trajectory.
        """
        segments = self._determine_trajectory_segments_with_max_velocity(
            start_time=start_time,
            curr_pos=curr_pos,
            curr_vel=curr_vel,
            target_position=target_position,
            max_vel=max_vel,
            accel=accel,
        )

        if segments[2].start_time <= segments[1].start_time:
            segments = self._determine_trajectory_segments_without_max_velocity(
                start_time=start_time,
                curr_pos=curr_pos,
                curr_vel=curr_vel,
                target_position=target_position,
                accel=accel,
            )

        return segments

    def _determine_trajectory_segments_with_max_velocity(
        self,
        start_time: float,
        curr_pos: float,
        curr_vel: float,
        target_position: float,
        max_vel: float,
        accel: float,
    ) -> list[TrajectorySegment]:
        """Determine the trajectory segments for the case where the maximum velocity can be reached.

        Parameters
        ----------
        start_time : `float`
            The start time [sec].
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        target_position : `float`
            The target position [any unit].
        max_vel : `float`
            The maximum velocity [any unit/sec].
        accel : `float`
            The acceleration [any unit/sec^2].

        Returns
        -------
        `list`[`TrajectorySegment`]
            The segments that form the entire trajectory.
        """
        time_to_max_vel = (max_vel - curr_vel) / accel
        pos_at_max_vel, vel_at_max_vel = accelerated_pos_and_vel(curr_pos, curr_vel, accel, time_to_max_vel)
        time_needed_to_stop_from_max_vel = max_vel / accel
        position_to_start_stopping, _ = accelerated_pos_and_vel(
            target_position, 0.0, -accel, time_needed_to_stop_from_max_vel
        )
        time_to_start_stopping = time_to_max_vel + (position_to_start_stopping - pos_at_max_vel) / max_vel

        return [
            TrajectorySegment(
                start_time=start_time,
                start_position=curr_pos,
                start_velocity=curr_vel,
                acceleration=accel,
            ),
            TrajectorySegment(
                start_time=start_time + time_to_max_vel,
                start_position=pos_at_max_vel,
                start_velocity=vel_at_max_vel,
                acceleration=0.0,
            ),
            TrajectorySegment(
                start_time=start_time + time_to_start_stopping,
                start_position=position_to_start_stopping,
                start_velocity=vel_at_max_vel,
                acceleration=-accel,
            ),
            TrajectorySegment(
                start_time=start_time + time_to_start_stopping + time_needed_to_stop_from_max_vel,
                start_position=target_position,
                start_velocity=0.0,
                acceleration=0.0,
            ),
        ]

    def _determine_trajectory_segments_without_max_velocity(
        self,
        start_time: float,
        curr_pos: float,
        curr_vel: float,
        target_position: float,
        accel: float,
    ) -> list[TrajectorySegment]:
        """Determine the trajectory segments for the case where the maximum velocity cannot be reached.

        Parameters
        ----------
        start_time : `float`
            The start time [sec].
        curr_pos : `float`
            The current position [any unit].
        curr_vel : `float`
            The current velocity [any unit/sec].
        target_position : `float`
            The target position [any unit].
        max_vel : `float`
            The maximum velocity [any unit/sec].
        accel : `float`
            The acceleration [any unit/sec^2].

        Returns
        -------
        `list`[`TrajectorySegment`]
            The segments that form the entire trajectory.
        """
        # Compute the time at which the velocity for the current motion is zero.
        # v0 + a*t = 0 <=> t = -v0/a
        t_zero_speed = -curr_vel / accel

        if t_zero_speed >= 0.0:
            # Compute the position at t_zero_speed.
            # p_zero_speed = p0 + v0*t_zero_speed + a*t_zero_speed**2/2.0
            p_zero_speed, _ = accelerated_pos_and_vel(curr_pos, curr_vel, accel, t_zero_speed)

            # Halfway between p_zero_speed and target is where the velocity needs to start decreasing.
            p_halfway = (p_zero_speed + target_position) / 2.0

            # Compute the time of p_halfway using numpy.
            # p_halfway = p0 + v0*t + a*t**2/2.0 <=> (a/2.0)*t**2 + (v0)*t + (p0 - p_halfway) = 0
            roots = np.roots([accel / 2.0, curr_vel, curr_pos - p_halfway])

            # If more than one root is found, take the largest one since the smallest one is usually negative,
            # and we are only interested in events in the future.
            t0 = np.max(roots)

            max_vel = curr_vel + t0 * accel

            return [
                TrajectorySegment(
                    start_time=start_time,
                    start_position=curr_pos,
                    start_velocity=curr_vel,
                    acceleration=accel,
                ),
                TrajectorySegment(
                    start_time=start_time + t0,
                    start_position=p_halfway,
                    start_velocity=max_vel,
                    acceleration=-accel,
                ),
                TrajectorySegment(
                    start_time=start_time + t0 + t0,
                    start_position=target_position,
                    start_velocity=0.0,
                    acceleration=0.0,
                ),
            ]
        else:
            time_needed_to_stop_from_v0 = abs(curr_vel / accel)
            position_to_start_stopping, _ = accelerated_pos_and_vel(
                target_position, 0.0, -accel, time_needed_to_stop_from_v0
            )

            # Halfway between p_zero_speed and target is where the velocity needs to start decreasing.
            p_halfway = (position_to_start_stopping + curr_pos) / 2.0

            # Compute the time of p_halfway using numpy.
            # p_halfway = p0 + v0*t + a*t**2/2.0 <=> (a/2.0)*t**2 + (v0)*t + (p0 - p_halfway) = 0
            roots = np.roots([accel / 2.0, curr_vel, curr_pos - p_halfway])

            # If more than one root is found, take the largest one since the smallest one is usually negative,
            # and we are only interested in events in the future.
            t0 = np.max(roots)

            max_vel = curr_vel + t0 * accel

            return [
                TrajectorySegment(
                    start_time=start_time,
                    start_position=curr_pos,
                    start_velocity=curr_vel,
                    acceleration=accel,
                ),
                TrajectorySegment(
                    start_time=start_time + t0,
                    start_position=p_halfway,
                    start_velocity=max_vel,
                    acceleration=-accel,
                ),
                TrajectorySegment(
                    start_time=start_time + t0 + t0 + time_needed_to_stop_from_v0,
                    start_position=target_position,
                    start_velocity=0.0,
                    acceleration=0.0,
                ),
            ]

    def consolidate_segments(self) -> None:
        """Consolidate the segments.

        Any segements that have the same acceleration as its predecessor is removed.
        """
        accel = self.segments[0].acceleration
        indices_of_segments_to_remove: list[int] = []
        for i in range(1, len(self.segments)):
            if math.isclose(self.segments[i].acceleration, accel):
                indices_of_segments_to_remove.append(i)
                accel = self.segments[i].acceleration
        for i in reversed(indices_of_segments_to_remove):
            self.segments.pop(i)
