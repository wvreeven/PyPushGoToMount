__all__ = ["EmulatedPlateSolver"]

from astropy.coordinates import SkyCoord  # type: ignore

from ..plate_solver import BasePlateSolver


class EmulatedPlateSolver(BasePlateSolver):
    async def solve(self) -> SkyCoord:
        raise RuntimeError("Exception thrown on purpose.")
