__all__ = ["PlateSolver"]

import asyncio
import logging
import math

import tetra3  # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
from PIL import Image

from ..camera import BaseCamera
from ..datetime_util import DatetimeUtil
from ..my_math import get_skycoord_from_ra_dec
from .base_plate_solver import BasePlateSolver

# Max FOV error [deg].
FOV_MAX_ERROR = 0.25
FOV_FACTOR = 202.265 / 60 / 60

# Solver timout [s].
SOLVER_TIMEOUT = 0.2


class PlateSolver(BasePlateSolver):
    """Plate solver that uses tetra3 to solve images."""

    def __init__(
        self,
        camera: BaseCamera,
        focal_length: float,
        log: logging.Logger,
    ) -> None:
        super().__init__(camera=camera, focal_length=focal_length, log=log)
        self.t3 = tetra3.Tetra3(load_database="asi120mm_database")

        self.center = get_skycoord_from_ra_dec(0.0, 0.0)
        self.previous_center = self.center

        self.fov_estimate = 0.0

    async def solve(self) -> SkyCoord:
        """Take an image and solve it.

        Returns
        -------
        SkyCoord
            The RA and Dec of the center of the image taken.

        Raises
        ------
        RuntimeError
            In case no image can be taken or solving it fails.
        """
        self.log.debug("Start solve.")
        start = DatetimeUtil.get_timestamp()
        if math.isclose(self.fov_estimate, 0.0):
            # Estimate of the size of the field of view [deg].
            max_img_size = max(self.camera.img_width, self.camera.img_height)
            self.fov_estimate = max_img_size * self.camera.pixel_size * FOV_FACTOR / self.focal_length
            self.log.info(f"{self.camera.img_width=}, {self.camera.img_height=}, {self.focal_length=}")

        self.log.debug(f"{self.fov_estimate=}")

        img_start = DatetimeUtil.get_timestamp()
        img = await self.get_image()
        img_end = DatetimeUtil.get_timestamp()
        self.log.debug(f"Async get_image took {img_end - img_start} s.")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._blocking_solve, img),
                timeout=SOLVER_TIMEOUT,
            )
        except Exception:
            self.center = self.previous_center
        finally:
            end = DatetimeUtil.get_timestamp()
            self.log.debug(f"Solving took {end - start} s.")
        return self.center

    def _blocking_solve(self, img: Image.Image) -> None:
        self.previous_center = self.center
        start = DatetimeUtil.get_timestamp()
        centroids = tetra3.get_centroids_from_image(image=img)
        end = DatetimeUtil.get_timestamp()
        self.log.debug(f"Centroids {end - start} s.")
        start = DatetimeUtil.get_timestamp()
        result = self.t3.solve_from_centroids(
            centroids,
            (img.height, img.width),
            fov_estimate=self.fov_estimate,
            fov_max_error=FOV_MAX_ERROR,
        )
        end = DatetimeUtil.get_timestamp()
        self.log.debug(f"Solve from centroids took {end - start} s.")
        self.center = get_skycoord_from_ra_dec(result["RA"], result["Dec"])
        self.fov_estimate = result["FOV"]
