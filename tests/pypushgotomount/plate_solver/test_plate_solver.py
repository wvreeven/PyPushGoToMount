import logging
import math
import pathlib
from unittest import IsolatedAsyncioTestCase

import numpy as np
import pypushgotomount
from PIL import Image

DATA_DIR = pathlib.Path(__file__).parents[1] / "test_data"


class TestPlateSolver(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        log = logging.getLogger(type(self).__name__)
        self.camera = pypushgotomount.emulation.EmulatedCamera(log=log)
        self.plate_solver = pypushgotomount.plate_solver.PlateSolver(
            camera=self.camera, focal_length=25.0, log=log
        )

    async def test_plate_solver(self) -> None:
        sky_coord = await self.plate_solver.solve()
        assert math.isclose(sky_coord.ra.deg, 0.0)
        assert math.isclose(sky_coord.dec.deg, 0.0)

    async def test_plate_solver_with_existing_image(self) -> None:
        self.camera.get_image = self.get_image  # type: ignore
        self.camera.img_width = 1280
        self.camera.img_height = 960
        self.camera.pixel_size = 3.76
        sky_coord = await self.plate_solver.solve()
        print(f"{sky_coord.ra.deg=}, {sky_coord.dec.deg=}")
        assert math.isclose(sky_coord.ra.deg, 271.20949435)
        assert math.isclose(sky_coord.dec.deg, 3.229427645)

    async def get_image(self) -> np.ndarray:
        img = Image.open(DATA_DIR / "test_image.png")
        return np.array(img)
