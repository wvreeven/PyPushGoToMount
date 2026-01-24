__all__ = [
    "AlignmentHandler",
    "AlignmentPoint",
    "AlignmentTriplet",
    "TelescopeAltAzFrame",
    "add_telescope_frame_transforms",
]

import asyncio
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
from astropy.coordinates import (
    AltAz,
    BaseCoordinateFrame,
    EarthLocationAttribute,
    RepresentationMapping,
    SkyCoord,
    SphericalRepresentation,
    StaticMatrixTransform,
    TimeAttribute,
    UnitSphericalRepresentation,
    frame_transform_graph,
)
from astropy.coordinates.matrix_utilities import matrix_transpose

from ..enums import IDENTITY, MILLISECOND


class TelescopeAltAzFrame(BaseCoordinateFrame):
    """Telescope coordinate frame."""

    default_representation = UnitSphericalRepresentation

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "az"),
            RepresentationMapping("lat", "alt"),
        ]
    }

    location = EarthLocationAttribute()
    obstime = TimeAttribute()


@dataclass(frozen=True)
class AlignmentPoint:
    """Class representing an alignment point.

    An alignment point consists of two points in the sky. One point is the actual [azimuth, altitude] and the
    other point is the [azimuth, altitude] where the telescope is pointing at. The difference is because of
    the telescope not havng been placed exactly horizontally.

    Parameters
    ----------
    altaz : `SkyCoord`
        The actual [azimuth, altitude].
    telescope : `SkyCoord`
        The telescope [azimuth, altitude].
    """

    altaz: SkyCoord
    telescope: SkyCoord

    def __str__(self) -> str:
        return (
            f"AltAz[az={self.altaz.az.deg}, alt={self.altaz.alt.deg}], "
            f"Telescope[az={self.telescope.az.deg}, alt={self.telescope.alt.deg}]"
        )

    def __repr__(self) -> str:
        return f"AlignmentPoint[{self}]"


@dataclass
class AlignmentTriplet:
    """A triplet of `AlignmentPoint`s.

    This is a utility class that is used to compute a transformation matrix for coordinate transformations
    between the actual and the telescope coordinate frames.

    Parameters
    ----------
    one : `AlignmentPoint`
        The first AlignmentPoint.
    two : `AlignmentPoint`
        The second AlignmentPoint.
    three : `AlignmentPoint`
        The third AlignmentPoint.
    """

    one: AlignmentPoint
    two: AlignmentPoint
    three: AlignmentPoint

    def altaz_as_altaz(self) -> SkyCoord:
        """Return the altaz coordinates of the three `AlignmentPoint`s as a single `SkyCoord`.

        Returns
        -------
        SkyCoord
            The altaz coordinates as a single `SkyCoord`.
        """
        return SkyCoord(
            [self.one.altaz.az, self.two.altaz.az, self.three.altaz.az],
            [self.one.altaz.alt, self.two.altaz.alt, self.three.altaz.alt],
        )

    def telescope_as_altaz(self) -> SkyCoord:
        """Return the telescope coordinates of the three `AlignmentPoint`s as a single `SkyCoord`.

        Returns
        -------
        SkyCoord
            The telescope coordinates as a single `SkyCoord`.
        """
        return SkyCoord(
            [self.one.telescope.az, self.two.telescope.az, self.three.telescope.az],
            [self.one.telescope.alt, self.two.telescope.alt, self.three.telescope.alt],
        )


class AlignmentHandler:
    """Utility class for handling the alignment of a telescope.

    A three-star alignment is assumed to be done. The observer is free to select the three stars to perform
    the alignment with. If after the three-star alignment the telescope is synced with more objects, they get
    added to the alignment handler so an improved transformation matrix can be computed.

    Attributes
    ----------
    matrix : `np.ndarray`
        The numpy array representing the transformation matrix.
    inv_matrix : `np.ndarray`
        The numpy array representing the inverse transformation matrix.
    """

    def __init__(self) -> None:
        self._alignment_data: list[AlignmentPoint] = []
        self.matrix = IDENTITY
        self.is_aligned = False
        add_telescope_frame_transforms(self.matrix)

    async def add_alignment_position(self, altaz: SkyCoord, telescope: SkyCoord) -> None:
        """Add an alignment point and compute the alignment matrix.

        Parameters
        ----------
        altaz : `SkyCoord`
            The actual [azimuth, altitude].
        telescope : `SkyCoord`
            The telescope [azimuth, altitude].
        """
        await asyncio.sleep(MILLISECOND)
        self._alignment_data.append(AlignmentPoint(altaz=altaz, telescope=telescope))
        await self._compute_transformation_matrix()

    async def _compute_transformation_matrix(self) -> None:
        """Compute the transformation matrix between the altaz and telescope coordinates.

        The transformation matrix only is computed if at least three alignment points have been added. If more
        than three alignment points have been added, the transformation matrix is computed for all unique
        combinations of three alignment points.

        If two or more alignment points are very close together, the resulting transformation matrix will
        contain one or more NaN values. In that case, the transformation matrix gets discarded. The mean
        matrix then is computed over the remaining matrices.

        See https://stackoverflow.com/a/27547597/22247307
        """
        await asyncio.sleep(MILLISECOND)
        transformation_matrices: list[np.ndarray] = []
        alignment_triplets = [
            AlignmentTriplet(one=triplet[0], two=triplet[1], three=triplet[2])
            for triplet in combinations(self._alignment_data, 3)
        ]
        for alignment_triplet in alignment_triplets:
            altaz_coords = alignment_triplet.altaz_as_altaz()
            telescope_coords = alignment_triplet.telescope_as_altaz()

            altaz_car = np.transpose(np.array(altaz_coords.cartesian.xyz.value))
            telescope_car = np.transpose(np.array(telescope_coords.cartesian.xyz.value))

            matrix = np.dot(np.linalg.inv(telescope_car), altaz_car)
            if not np.any(np.isnan(matrix)):
                transformation_matrices.append(matrix)

        if len(transformation_matrices) == 0:
            self.matrix = IDENTITY
        elif len(transformation_matrices) == 1:
            self.matrix = transformation_matrices[0]
            self.is_aligned = True
        else:
            # Compute the mean.
            self.matrix = np.mean(transformation_matrices, axis=0)
            self.is_aligned = True

        add_telescope_frame_transforms(self.matrix)

    async def get_telescope_coords_from_altaz(self, altaz_coord: SkyCoord) -> SkyCoord:
        await asyncio.sleep(MILLISECOND)
        telescope_coord = altaz_coord.transform_to(TelescopeAltAzFrame)
        return telescope_coord

    async def get_altaz_from_telescope_coords(self, telescope_coord: SkyCoord) -> SkyCoord:
        await asyncio.sleep(MILLISECOND)
        altaz_coord = telescope_coord.transform_to(AltAz)
        return altaz_coord


def add_telescope_frame_transforms(matrix: npt.ArrayLike) -> None:
    to_telescope = StaticMatrixTransform(matrix=matrix, fromsys=AltAz, tosys=TelescopeAltAzFrame)
    to_altaz = StaticMatrixTransform(
        matrix=matrix_transpose(matrix),
        fromsys=TelescopeAltAzFrame,
        tosys=AltAz,
    )
    frame_transform_graph.add_transform(AltAz, TelescopeAltAzFrame, to_telescope)
    frame_transform_graph.add_transform(TelescopeAltAzFrame, AltAz, to_altaz)
