__all__ = ["BaseCamera"]

import abc
import logging

import numpy as np


class BaseCamera(abc.ABC):
    """Interface for all camera implementations."""

    def __init__(self, log: logging.Logger) -> None:
        self.log = log.getChild(type(self).__name__)
        self.camera_id = 0
        self.img_width = 0
        self.img_height = 0
        self.pixel_size = 0.0

    @abc.abstractmethod
    async def open(self) -> None:
        """Open the camera."""
        raise NotImplementedError

    @abc.abstractmethod
    async def start_imaging(self) -> None:
        """Close the camera."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_image(self) -> np.ndarray:
        """Get the latest image.

        Returns
        -------
        numpy.ndarray
            The image as a numpy array.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def stop_imaging(self) -> None:
        """Close the camera."""
        raise NotImplementedError
