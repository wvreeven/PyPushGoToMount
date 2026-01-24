import math
import unittest

import astropy.units as u
import numpy as np
import pylx200mount
from astropy.coordinates import AltAz, SkyCoord


class TestAlignmentHandler(unittest.IsolatedAsyncioTestCase):
    # Rotation of 1º around the x-axis.
    angle = math.radians(1.0)
    matrix = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    async def test_telescope_frame(self) -> None:
        now = pylx200mount.DatetimeUtil.get_timestamp()

        pylx200mount.alignment.add_telescope_frame_transforms(self.matrix)

        coo = SkyCoord(az=[0.0, 90.0, 120.0] * u.deg, alt=[41.3, 86.8, 77.9] * u.deg, frame=AltAz)
        expected = SkyCoord(az=[1.0, 91.0, 121.0] * u.deg, alt=[41.3, 86.8, 77.9] * u.deg, frame=AltAz)
        tel_coo = coo.transform_to(pylx200mount.alignment.TelescopeAltAzFrame)
        assert np.all(np.isclose(tel_coo.az, expected.az))
        assert np.all(np.isclose(tel_coo.alt, expected.alt))
        coo2 = tel_coo.transform_to(AltAz)
        assert np.all(np.isclose(coo.az, coo2.az))
        assert np.all(np.isclose(coo.alt, coo2.alt))

        altaz = await pylx200mount.my_math.get_skycoord_from_alt_az(az=0.0, alt=41.3, timestamp=now)
        tel_coo = altaz.transform_to(pylx200mount.alignment.TelescopeAltAzFrame)
        assert math.isclose(tel_coo.az.deg, expected[0].az.deg)
        assert math.isclose(tel_coo.alt.deg, expected[0].alt.deg)

    async def test_create_matrix(self) -> None:
        coo = SkyCoord([0.0, 90.0, 120.0] * u.deg, [41.3, 86.8, 77.9] * u.deg)
        tel_coo = SkyCoord(
            [1.0, 91.0, 121.0] * u.deg,
            [41.3, 86.8, 77.9] * u.deg,
            frame=pylx200mount.alignment.TelescopeAltAzFrame,
        )

        coo_car = np.transpose(np.array(coo.cartesian.xyz.value))
        tel_coo_car = np.transpose(np.array(tel_coo.cartesian.xyz.value))

        m = np.dot(np.linalg.inv(tel_coo_car), coo_car)
        assert np.all(np.isclose(m, self.matrix))

    async def test_create_matrix_using_alignment_points(self) -> None:
        now = pylx200mount.DatetimeUtil.get_timestamp()
        ap1 = pylx200mount.alignment.AlignmentPoint(
            altaz=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=0.0,
                alt=41.3,
                timestamp=now,
            ),
            telescope=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=1.0,
                alt=41.3,
                timestamp=now,
                frame=pylx200mount.alignment.TelescopeAltAzFrame,
            ),
        )
        ap2 = pylx200mount.alignment.AlignmentPoint(
            altaz=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=90.0,
                alt=41.3,
                timestamp=now,
            ),
            telescope=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=91.0,
                alt=41.3,
                timestamp=now,
                frame=pylx200mount.alignment.TelescopeAltAzFrame,
            ),
        )
        ap3 = pylx200mount.alignment.AlignmentPoint(
            altaz=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=120.0,
                alt=41.3,
                timestamp=now,
            ),
            telescope=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=121.0,
                alt=41.3,
                timestamp=now,
                frame=pylx200mount.alignment.TelescopeAltAzFrame,
            ),
        )
        aps = pylx200mount.alignment.AlignmentTriplet(ap1, ap2, ap3)

        coo = aps.altaz_as_altaz()
        tel_coo = aps.telescope_as_altaz()

        coo_car = np.transpose(np.array(coo.cartesian.xyz.value))
        tel_coo_car = np.transpose(np.array(tel_coo.cartesian.xyz.value))

        m = np.dot(np.linalg.inv(tel_coo_car), coo_car)
        assert np.all(np.isclose(m, self.matrix))

    async def test_alignment_handler(self) -> None:
        now = pylx200mount.DatetimeUtil.get_timestamp()
        ah = pylx200mount.alignment.AlignmentHandler()
        altaz = await pylx200mount.my_math.get_skycoord_from_alt_az(az=0.0, alt=41.3, timestamp=now)
        await ah.add_alignment_position(
            altaz,
            telescope=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=1.0,
                alt=41.3,
                timestamp=now,
                frame=pylx200mount.alignment.TelescopeAltAzFrame,
            ),
        )
        await ah.add_alignment_position(
            altaz=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=90.0,
                alt=41.3,
                timestamp=now,
            ),
            telescope=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=91.0,
                alt=41.3,
                timestamp=now,
                frame=pylx200mount.alignment.TelescopeAltAzFrame,
            ),
        )
        await ah.add_alignment_position(
            altaz=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=120.0,
                alt=41.3,
                timestamp=now,
            ),
            telescope=await pylx200mount.my_math.get_skycoord_from_alt_az(
                az=121.0,
                alt=41.3,
                timestamp=now,
                frame=pylx200mount.alignment.TelescopeAltAzFrame,
            ),
        )
        assert np.all(np.isclose(ah.matrix, self.matrix))

        tel = await ah.get_telescope_coords_from_altaz(altaz)
        assert math.isclose(tel.az.deg, 1.0)
        assert math.isclose(tel.alt.deg, 41.3)

        coo = await ah.get_altaz_from_telescope_coords(tel)
        coo_az = coo.az.wrap_at(180 * u.deg)
        assert math.isclose(coo_az.deg, 0.0, abs_tol=1.0e-10)
        assert math.isclose(coo.alt.deg, 41.3)
