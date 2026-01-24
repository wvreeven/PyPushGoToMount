import asyncio
import importlib
import logging
import math
import pathlib
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import astropy.units as u
import pylx200mount
from astropy.coordinates import Angle, SkyCoord
from pylx200mount import datetime_util

AZ_DEVIATION = Angle(1.0 * u.deg)
ALT_DEVIATION = Angle(0.0 * u.deg)

# AltAz tolerance converted from arcminutes to degrees.
ALTAZ_TOLERANCE = u.arcminute.to(u.deg, 10.0)

# Directory holding configuration files for unit tests.
CONFIG_DIR = pathlib.Path(__file__).parents[0] / "test_data"

# AltAz of second alignment point.
AZ_SECOND_POINT = Angle(45.0 * u.deg)
ALT_SECOND_POINT = Angle(60.0 * u.deg)
# AltAz of third alignment point.
AZ_THIRD_POINT = Angle(90.0 * u.deg)
ALT_THIRD_POINT = Angle(50.0 * u.deg)

# Start position of the motors.
MOTOR_START_POSITION = 0.0


class TestLx200Mount(unittest.IsolatedAsyncioTestCase):
    async def test_lx200_mount(self) -> None:
        self.log = logging.getLogger(type(self).__name__)
        self.now = datetime.now(timezone(timedelta(hours=+1), "CET"))
        config_file = CONFIG_DIR / "config_emulated_motors_only.json"

        # RaDec of Polaris for first alignment point.
        polaris = await pylx200mount.my_math.get_skycoord_from_ra_dec(37.95456067, 89.26410897)

        importlib.reload(datetime_util)

        self.expected_alt_offset = 0.0
        self.expected_az_offset = 0.0

        altaz_second_point = await pylx200mount.my_math.get_skycoord_from_alt_az(
            ALT_SECOND_POINT.deg, AZ_SECOND_POINT.deg, timestamp=self.now.timestamp()
        )
        radec_second_point = await pylx200mount.my_math.get_radec_from_altaz(altaz_second_point)
        altaz_third_point = await pylx200mount.my_math.get_skycoord_from_alt_az(
            ALT_THIRD_POINT.deg, AZ_THIRD_POINT.deg, timestamp=self.now.timestamp()
        )
        radec_third_point = await pylx200mount.my_math.get_radec_from_altaz(altaz_third_point)

        with mock.patch("pylx200mount.controller.utils.CONFIG_FILE", config_file):
            async with pylx200mount.LX200Mount(run_forever=False) as self.lx200_mount:
                self.reader, self.writer = await asyncio.open_connection(host="localhost", port=11880)
                await self.send_start_commands()

                self.motor_controller_alt = self.lx200_mount.responder.mount_controller.motor_controller_alt
                self.motor_controller_az = self.lx200_mount.responder.mount_controller.motor_controller_az

                self.expected_alt_offset = ALT_DEVIATION.deg
                self.expected_az_offset = AZ_DEVIATION.deg

                await self.set_and_assert_position(polaris)
                await self.set_and_assert_position(radec_second_point)
                await self.set_and_assert_position(radec_third_point)

                await self.lx200_mount.stop()

    async def send_start_commands(self) -> None:
        utcoffset = self.now.utcoffset()
        assert utcoffset is not None
        utcoffset_hours = utcoffset.total_seconds() / 60.0 / 60.0
        now_date = self.now.strftime("%m/%d/%y")
        now_time = self.now.strftime("%H:%M:%S")
        for st, expected in [
            (f":{pylx200mount.CommandName.St}+40*30#", "1"),
            (f":{pylx200mount.CommandName.S_LOWER_G}003*53#", "1"),
            (f":{pylx200mount.CommandName.S_UPPER_G}{-utcoffset_hours:04.1f}#", "1"),
            (f":{pylx200mount.CommandName.SL}{now_time}#", "1"),
            (f":{pylx200mount.CommandName.SC}{now_date}#", "1"),
            (f":{pylx200mount.CommandName.U}#", None),
            (f":{pylx200mount.CommandName.RG}#", None),
        ]:
            self.writer.write(st.encode())
            await self.writer.drain()
            if expected is not None:
                data = await self.reader.read(1)
                assert data.decode() == expected
                if st.startswith(f":{pylx200mount.CommandName.SC}"):
                    await self.reader.readuntil(b"#")
                    await self.reader.readuntil(b"#")

        await self.assert_position(
            expected=await pylx200mount.my_math.get_skycoord_from_alt_az(
                alt=MOTOR_START_POSITION,
                az=MOTOR_START_POSITION,
                timestamp=self.now.timestamp(),
            )
        )

    async def set_and_assert_position(self, position: SkyCoord) -> None:
        assert self.motor_controller_alt is not None
        assert self.motor_controller_az is not None
        altaz = await pylx200mount.my_math.get_altaz_from_radec(position, self.now.timestamp())
        motor_altaz = altaz.spherical_offsets_by(AZ_DEVIATION, ALT_DEVIATION)
        self.motor_controller_alt._position = (
            motor_altaz.alt.deg / self.motor_controller_alt._conversion_factor.deg
        )
        self.motor_controller_az._position = (
            motor_altaz.az.deg / self.motor_controller_az._conversion_factor.deg
        )
        await self.assert_position(expected=altaz)

        # Send align commands.
        ra_str = position.ra.to_string(u.hour, precision=0, pad=True, sep=":")
        dec_str = position.dec.to_string(u.deg, precision=0, pad=True, alwayssign=True, sep=("*", ":"))
        self.writer.write(f":{pylx200mount.CommandName.Sr}{ra_str}#".encode())
        await self.writer.drain()
        self.writer.write(f":{pylx200mount.CommandName.Sd}{dec_str}#".encode())
        await self.writer.drain()
        self.writer.write(f":{pylx200mount.CommandName.CM}#".encode())
        await self.writer.drain()
        await self.reader.readuntil(b"#")

    async def assert_position(self, expected: SkyCoord) -> None:
        self.lx200_mount.responder.mount_controller.position_event.clear()
        await self.lx200_mount.responder.mount_controller.position_event.wait()

        self.writer.write(f":{pylx200mount.CommandName.GR}#".encode())
        await self.writer.drain()
        ra = (await self.reader.readuntil(b"#")).decode().strip("#")

        self.writer.write(f":{pylx200mount.CommandName.GD}#".encode())
        await self.writer.drain()
        de = (await self.reader.readuntil(b"#")).decode().strip("#").replace("*", ":").replace("'", ":")

        ra_dec = await pylx200mount.my_math.get_skycoord_from_ra_dec_str(ra, de)
        self.log.debug(f"RaDec = {ra_dec.to_string('hmsdms')}")

        altaz = await pylx200mount.my_math.get_altaz_from_radec(ra_dec, timestamp=self.now.timestamp())
        self.log.debug(f"AltAz = {altaz.to_string('dms')}")
        self.log.debug(f"Expected AltAz = {expected.to_string('dms')}")

        daz = (altaz.az - expected.az).wrap_at(180 * u.deg).deg * math.cos(altaz.alt.rad)
        dalt = altaz.alt.deg - expected.alt.deg
        assert math.isclose(daz, self.expected_az_offset, abs_tol=ALTAZ_TOLERANCE)
        assert math.isclose(dalt, self.expected_alt_offset, abs_tol=ALTAZ_TOLERANCE)
