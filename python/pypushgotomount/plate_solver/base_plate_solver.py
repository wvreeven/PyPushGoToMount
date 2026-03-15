__all__ = ["BasePlateSolver"]

import abc
import logging

from astropy.coordinates import SkyCoord  # type: ignore
from PIL import Image

from ..camera import BaseCamera


class BasePlateSolver(abc.ABC):
    """Base class for plate solvers."""

    def __init__(self, camera: BaseCamera, focal_length: float, log: logging.Logger) -> None:
        self.log = log.getChild(type(self).__name__)
        self.camera = camera
        self.focal_length = focal_length

    async def open_camera(self) -> None:
        """Open the camera and make sure it uses the full sensor."""
        self.log.debug("Opening camera.")
        await self.camera.open()

    async def start_imaging(self) -> None:
        """Instruct the camera to start imaging."""
        await self.camera.start_imaging()

    async def get_image(self) -> Image.Image:
        """Get the latest image from the camera.

        Returns
        -------
        Image
            The image taken with the camera in PIL Image format.
        """
        data = await self.camera.get_image()
        img = Image.fromarray(data)
        return img

    async def stop_imaging(self) -> None:
        """Instruct the camera to stop imaging."""
        await self.camera.stop_imaging()

    @abc.abstractmethod
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
        raise NotImplementedError
