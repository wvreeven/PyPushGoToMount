__all__ = ["DatetimeUtil"]

from datetime import datetime, timedelta, timezone


class DatetimeUtil:
    """Utility class to work with datetime instances.

    Uisng a computer without internet connection and without a real time clock may result in inaccurate
    pointing due to time differences between the hardware clock and the world. This utility class aims to
    solve that for this project.
    """

    _dt = datetime.now().astimezone()
    assert _dt is not None
    utcoffset = _dt.utcoffset()
    assert utcoffset is not None

    delta = timedelta()
    tz = timezone(utcoffset)

    @classmethod
    def get_datetime(cls) -> datetime:
        """Convenience method to retrieve the current timezone datetime.

        This takes both the timezone (either taken from the computer or set by the planetarium software) and
        the timedelta (taken from the time and date set by the planetarium software) into account.

        Unit tests may replace this function for better time control.

        Returns
        -------
        `float`
            The current timezone datetime.
        """
        return datetime.now().astimezone(DatetimeUtil.tz) + DatetimeUtil.delta

    @classmethod
    def get_datetime_at_timestamp(cls, timestamp: float) -> datetime:
        """Convenience method to retrieve the timezone datetime at the provided timestamp.

        This takes both the timezone (either taken from the computer or set by the planetarium software) and
        the timestamp into account.

        Unit tests may replace this function for better time control.

        Parameters
        ----------
        timestamp : `float`
            The current timestamp.

        Returns
        -------
        `float`
            The timezone datetime at thge provided timestamp.
        """
        return datetime.fromtimestamp(timestamp, DatetimeUtil.tz)

    @classmethod
    def get_timestamp(cls) -> float:
        """Convenience method to retrieve the current time stamp.

        Returns
        -------
        `float`
            The current timezone time as timestamp.
        """
        return DatetimeUtil.get_datetime().timestamp()

    @classmethod
    def get_timestamp_from_timestamp(cls, timestamp: float) -> float:
        """Convenience method to retrieve the time stamp for the current timezone.

        Parameters
        ----------
        timestamp : `float`
            The current timestamp.

        Returns
        -------
        `float`
            The current timezone time as timestamp.
        """
        return datetime.fromtimestamp(timestamp, DatetimeUtil.tz).timestamp()

    @classmethod
    def set_datetime(cls, dt: datetime) -> None:
        """Set the date.

        The difference between the time reported by the hardware clock and reported by the planetarium
        software is determined and stored so the `get_datetime` method can use it. The timezone used by this
        class is also updated.

        Parameters
        ----------
        dt : `datetime`
            The datetime as set by the planetarium software.
        """
        # noinspection PyTypeChecker
        DatetimeUtil.delta = DatetimeUtil.get_datetime() - dt
        DatetimeUtil.utcoffset = dt.utcoffset()
        assert DatetimeUtil.utcoffset is not None
        DatetimeUtil.tz = timezone(DatetimeUtil.utcoffset)
