__all__ = [
    "Lx200CommandResponder",
    "REPLY_SEPARATOR",
    "UPDATING_PLANETARY_DATA1",
    "UPDATING_PLANETARY_DATA2",
]

import asyncio
import logging
from datetime import datetime

from astropy import units as u
from astropy.coordinates import Angle, Latitude, Longitude, SkyCoord

from ..datetime_util import DatetimeUtil
from ..enums import MILLISECOND, CommandName, CoordinatePrecision
from ..my_math.astropy_util import get_skycoord_from_ra_dec_str
from ..observing_location import get_observing_location, set_latitude, set_longitude
from .mount_controller import MountController

# Some replies are terminated by the hash symbol.
HASH = "#"

# Default reply.
DEFAULT_REPLY = "1"

# A slew to the target position is possible.
SLEW_POSSIBLE = "0"

# Multiple strings which get sent as a reply to the SC command.
UPDATING_PLANETARY_DATA1 = "Updating Planetary Data       " + HASH
UPDATING_PLANETARY_DATA2 = "                              " + HASH

# Separator used for multiple replies.
REPLY_SEPARATOR = "\n"


async def get_angle_as_lx200_string(
    angle: Angle, digits: int, coordinate_precision: CoordinatePrecision
) -> str:
    await asyncio.sleep(MILLISECOND)

    # Use signed_dms here because dms will have negative minutes and seconds!!!
    angle_dms = angle.signed_dms
    d = f"{angle_dms.sign * angle_dms.d:0{digits}.0f}"
    if coordinate_precision == CoordinatePrecision.HIGH:
        angle_str = f"{d}*{angle_dms.m:02.0f}'{angle_dms.s:02.0f}"
    else:
        m = angle_dms.m + (angle_dms.s / 60.0)
        angle_str = f"{d}*{m:02.0f}"
    return angle_str


class Lx200CommandResponder:
    """Implements the LX200 protocol.

    For more info, see the TelescopeProtocol PDF document in the misc/docs section and

    https://www.astro.louisville.edu/software/xmtel/archive/
        xmtel-indi-6.0/xmtel-6.0l/support/lx200/CommandSet.html
    """

    def __init__(self, log: logging.Logger) -> None:
        self.log = log.getChild(type(self).__name__)

        # Variables holding the status of the mount
        self.autoguide_speed = 0

        # Variables holding the target position
        self.target_ra = "0.0"
        self.target_dec = "0.0"

        self.mount_controller = MountController(log=self.log)

        # The received command. This is kept as a reference for the slews.
        self.cmd: str = ""
        # Dictionary of the functions to execute based on the received command.
        self.dispatch_dict = {
            CommandName.CM: (self.sync, False),
            CommandName.D: (self.get_distance_bars, False),
            CommandName.G_LOWER_C: (self.get_clock_format, False),
            CommandName.G_UPPER_C: (self.get_current_date, False),
            CommandName.GD: (self.get_dec, False),
            CommandName.G_LOWER_G: (self.get_current_site_longitude, False),
            CommandName.G_UPPER_G: (self.get_utc_offset, False),
            CommandName.GL: (self.get_local_time, False),
            CommandName.GM: (self.get_site_1_name, False),
            CommandName.GR: (self.get_ra, False),
            CommandName.G_LOWER_T: (self.get_current_site_latitude, False),
            CommandName.G_UPPER_T: (self.get_tracking_rate, False),
            CommandName.GVD: (self.get_firmware_date, False),
            CommandName.GVF: (self.get_firmware_name, False),
            CommandName.GVN: (self.get_firmware_number, False),
            CommandName.GVP: (self.get_telescope_name, False),
            CommandName.GVT: (self.get_firmware_time, False),
            CommandName.GW: (self.get_firmware_time, False),
            CommandName.H: (self.toggle_time_format, False),
            CommandName.Mn: (self.move_slew_in_direction, False),
            CommandName.Me: (self.move_slew_in_direction, False),
            CommandName.M_LOWER_S: (self.move_slew_in_direction, False),
            CommandName.M_UPPER_S: (self.move_slew, False),
            CommandName.Mw: (self.move_slew_in_direction, False),
            CommandName.Qn: (self.stop_slew, False),
            CommandName.Qe: (self.stop_slew, False),
            CommandName.Qs: (self.stop_slew, False),
            CommandName.Qw: (self.stop_slew, False),
            CommandName.Q_HASH: (self.stop_slew, False),
            CommandName.RC: (self.set_slew_rate, False),
            CommandName.RG: (self.set_slew_rate, False),
            CommandName.RM: (self.set_slew_rate, False),
            CommandName.RS: (self.set_slew_rate, False),
            CommandName.SC: (self.set_local_date, True),
            CommandName.Sd: (self.set_dec, True),
            CommandName.S_LOWER_G: (self.set_current_site_longitude, True),
            CommandName.S_UPPER_G: (self.set_utc_offset, True),
            CommandName.SL: (self.set_local_time, True),
            CommandName.Sr: (self.set_ra, True),
            CommandName.St: (self.set_current_site_latitude, True),
            CommandName.U: (self.set_coordinate_precision, False),
        }
        # The coordinate precision.
        self.coordinate_precision = CoordinatePrecision.LOW

        # Keep track of the timezone, time and date, so it can be passed on to DatetimeUtil.
        self._datetime_str = ""
        # Keep track of the mount position for quicker responses.
        self.ra_dec = SkyCoord(0.0, 0.0, unit="deg")

    async def start(self) -> None:
        """Start the responder."""
        self.log.info("Start called.")
        await self.mount_controller.start()

    async def stop(self) -> None:
        """Stop the responder."""
        self.log.info("Stop called.")
        await self.mount_controller.stop()

    async def get_ra(self) -> str:
        """Get the RA that the mount currently is pointing at."""
        self.ra_dec = await self.mount_controller.get_ra_dec()
        ra = self.ra_dec.ra
        hms = ra.hms
        if self.coordinate_precision == CoordinatePrecision.HIGH:
            ra_str = f"{hms.h:02.0f}:{hms.m:02.0f}:{hms.s:02.0f}"
        else:
            m = hms.m + (hms.s / 60.0)
            ra_str = f"{hms.h:02.0f}:{m:02.1f}"
        return ra_str + HASH

    async def set_ra(self, data: str) -> str:
        """Set the RA that the mount should slew to.

        Parameters
        ----------
        data: `str`
            A sexagemal representation of the RA, HH:mm:ss

        Returns
        -------
        DEFAULT_REPLY
            The default reply accoring to the LX200 command protocol.
        """
        await asyncio.sleep(MILLISECOND)
        self.log.debug(f"Setting RA to {data}")
        self.target_ra = data
        return DEFAULT_REPLY

    async def get_dec(self) -> str:
        """Get the DEC that the mount currently is pointing at."""
        dec_str = await get_angle_as_lx200_string(
            angle=self.ra_dec.dec,
            digits=2,
            coordinate_precision=self.coordinate_precision,
        )
        return dec_str + HASH

    async def set_dec(self, data: str) -> str:
        """Set the DEC that the mount should slew to.

        Parameters
        ----------
        data: `str`
            A sexagemal representation of the DEC, dd*mm:ss

        Returns
        -------
        DEFAULT_REPLY
            The default reply accoring to the LX200 command protocol.
        """
        await asyncio.sleep(MILLISECOND)
        self.log.debug(f"Setting DEC to {data}")
        self.target_dec = data
        return DEFAULT_REPLY

    async def get_clock_format(self) -> str:
        """Get the clock format: 12h or 24h. We will always use 24h."""
        await asyncio.sleep(MILLISECOND)
        return "(24)" + HASH

    async def get_tracking_rate(self) -> str:
        """Get the tracking rate of the mount."""
        # Return the sideral tracking frequency.
        await asyncio.sleep(MILLISECOND)
        return "60.1" + HASH

    async def get_utc_offset(self) -> str:
        """Get the UTC offset of the obsering site.

        The LX200 counts the number of hours that need to be added to get the UTC instead
        of the number of hours that the local time is ahead or behind of UTC. The
        difference is a minus symbol.
        """
        await asyncio.sleep(MILLISECOND)
        dt = DatetimeUtil.get_datetime()
        tz_info = dt.tzinfo
        assert tz_info is not None
        utc_offset = tz_info.utcoffset(dt)
        assert utc_offset is not None
        utc_offset_hours = -utc_offset.total_seconds() / 3600
        self.log.debug(f"UTC Offset = {utc_offset_hours}")
        return f"{utc_offset_hours:2.0f}" + HASH

    async def get_local_time(self) -> str:
        """Get the local time at the observing site."""
        await asyncio.sleep(MILLISECOND)
        current_dt = DatetimeUtil.get_datetime()
        return current_dt.strftime("%H:%M:%S") + HASH

    async def get_current_date(self) -> str:
        """Get the local date at the observing site."""
        await asyncio.sleep(MILLISECOND)
        current_dt = DatetimeUtil.get_datetime()
        return current_dt.strftime("%m/%d/%y") + HASH

    async def get_firmware_date(self) -> str:
        """Get the firmware date which is just a date that I made up."""
        await asyncio.sleep(MILLISECOND)
        return "Apr 05 2020" + HASH

    async def get_firmware_time(self) -> str:
        """Get the firmware time which is just a time that I made up."""
        await asyncio.sleep(MILLISECOND)
        return "18:00:00" + HASH

    async def get_firmware_number(self) -> str:
        """Get the firmware number which is just a number that I made up."""
        await asyncio.sleep(MILLISECOND)
        return "01.0" + HASH

    async def get_firmware_name(self) -> str:
        """Get the firmware name which is just a name that I made up."""
        await asyncio.sleep(MILLISECOND)
        return "Phidgets|A|43Eg|Apr 05 2020@18:00:00" + HASH

    async def get_telescope_name(self) -> str:
        """Get the mount name which is just a name that I made up."""
        await asyncio.sleep(MILLISECOND)
        return "Phidgets" + HASH

    async def get_current_site_latitude(self) -> str:
        """Get the latitude of the obsering site."""
        lat = get_observing_location().lat
        return (
            await get_angle_as_lx200_string(angle=lat, digits=2, coordinate_precision=CoordinatePrecision.LOW)
            + HASH
        )

    async def set_current_site_latitude(self, data: str) -> str:
        """Set the latitude of the obsering site."""
        self.log.debug(f"set_current_site_latitude received data {data}")
        if "*" in data:
            # SkySafari and AstroPlanner send the latitude in the form "deg*min".
            lat_deg, lat_min = data.split("*")
            set_latitude(Latitude(f"{lat_deg}d{lat_min}m"))
        else:
            # INDI sends the latitude in the form of a decimal value.
            set_latitude(Latitude(f"{data} degrees"))
        self.log.debug(
            f"Converted LX200 latitude {data} to internal latitude {get_observing_location().lat.deg}"
        )
        await self.mount_controller.location_updated()
        return DEFAULT_REPLY

    async def get_current_site_longitude(self) -> str:
        """Get the longitude of the obsering site.

        The LX200 protocol puts West longitudes positive while astropy puts East
        longitude positive, so we need to convert from the astropy longitude to the
        LX200 longitude.
        """
        await asyncio.sleep(MILLISECOND)
        longitude = get_observing_location().lon.to_string(unit=u.degree, sep=":", fields=2)
        if longitude[0] == "-":
            longitude = longitude[1:]
        else:
            longitude = "-" + longitude
        self.log.debug(
            f"Converted internal longitude "
            f"{get_observing_location().lon.to_string()} "
            f"to LX200 longitude {longitude}"
        )
        return longitude + HASH

    async def set_current_site_longitude(self, data: str) -> str:
        """Set the longitude of the obsering site.

        The LX200 protocol puts West longitudes positive while astropy puts East
        longitude positive, so we need to convert from the LX200 longitude to the
        astropy longitude.
        """
        self.log.debug(f"set_current_site_longitude received data {data}")
        if data[0] == "-":
            longitude = data[1:]
        else:
            longitude = "-" + data

        if "*" in data:
            # SkySafari and AstroPlanner send the longitude in the form "deg*min".
            lon_deg, lon_min = longitude.split("*")
            set_longitude(Latitude(f"{lon_deg}d{lon_min}m"))
        else:
            # INDI sends the longitude in the form of a decimal value.
            set_longitude(Longitude(f"{longitude} degrees"))
        self.log.debug(
            f"Converted LX200 longitude {data} to internal longitude {get_observing_location().lon.deg}"
        )
        await self.mount_controller.location_updated()
        return DEFAULT_REPLY

    async def get_site_1_name(self) -> str:
        """Get the name of the observing site."""
        await asyncio.sleep(MILLISECOND)
        return "My Site" + HASH

    async def set_slew_rate(self) -> None:
        """Set the slew rate to the commanded rate."""
        self.log.debug(f"Setting slew rate to value determined by command {self.cmd}")
        await self.mount_controller.set_slew_rate(cmd=self.cmd)

    async def move_slew(self) -> str:
        """Move the telescope at slew rate to the target position."""
        self.log.debug(f"Slewing to RaDec ({self.target_ra}, {self.target_dec}).")
        slew_possible = await self.mount_controller.slew_to(self.target_ra, self.target_dec)
        if slew_possible != "0":
            slew_possible = slew_possible + HASH
        return slew_possible

    async def move_slew_in_direction(self) -> None:
        """Move the telescope at slew rate in the commanded direction."""
        self.log.debug(f"Slewing in direction determined by cmd {self.cmd}")
        await self.mount_controller.slew_in_direction(cmd=self.cmd)

    async def stop_slew(self) -> None:
        """Stop the current slew."""
        self.log.debug("Stopping current slew.")
        await self.mount_controller.stop_slew()

    async def set_utc_offset(self, data: str) -> str:
        """Set the UTC offset.

        This is the first method to be called in sequence when the planetarium software sets the timezone,
        time and date.
        """
        await asyncio.sleep(MILLISECOND)
        self.log.debug(f"set_utc_offset received data {data}")
        utc_offset_hours = -float(data)
        self._datetime_str = f"{100 * utc_offset_hours:+05.0f}"
        return DEFAULT_REPLY

    async def set_local_time(self, data: str) -> str:
        """Set the local time.

        This is the second method to be called in sequence when the planetarium software sets the timezone,
        time and date.
        """
        await asyncio.sleep(MILLISECOND)
        self.log.debug(f"set_local_time received data {data}")
        self._datetime_str = data + self._datetime_str
        return DEFAULT_REPLY

    async def set_local_date(self, data: str) -> str:
        """Set the local date.

        This is the third method to be called in sequence when the planetarium software sets the timezone,
        time and date.
        """
        await asyncio.sleep(MILLISECOND)
        self.log.debug(f"set_local_date received data {data}")
        self._datetime_str = data + "T" + self._datetime_str
        dt = datetime.strptime(self._datetime_str, "%m/%d/%yT%H:%M:%S%z")
        DatetimeUtil.set_datetime(dt)
        # Two return strings are expected so here we separate them by a new
        # line character and will let the socket server deal with it.
        return (
            DEFAULT_REPLY
            + REPLY_SEPARATOR
            + UPDATING_PLANETARY_DATA1
            + REPLY_SEPARATOR
            + UPDATING_PLANETARY_DATA2
        )

    async def toggle_time_format(self) -> None:
        """Toggle the time format."""
        await asyncio.sleep(MILLISECOND)
        self.log.debug("toggle_time_format received.")

    async def set_coordinate_precision(self) -> None:
        """Set the coordinate precision."""
        await asyncio.sleep(MILLISECOND)
        self.log.debug("set_coordinate_precision")
        self.coordinate_precision = (
            CoordinatePrecision.LOW
            if self.coordinate_precision == CoordinatePrecision.HIGH
            else CoordinatePrecision.HIGH
        )

    async def sync(self) -> str:
        self.log.debug("sync received.")
        ra_dec = await get_skycoord_from_ra_dec_str(ra_str=self.target_ra, dec_str=self.target_dec)
        await self.mount_controller.set_ra_dec(ra_dec)
        return "RANDOM NAME" + HASH

    async def get_distance_bars(self) -> str:
        """Get the distance bars displayed on the hand controller.

        The hand controller doesn't exist in this case."""
        await asyncio.sleep(MILLISECOND)
        self.log.debug("get_distance_bars received.")
        return "0x7f" + HASH

    async def get_alignment_status(self) -> str:
        """Get the alignment status."""
        await asyncio.sleep(MILLISECOND)
        return "AN0" + HASH
