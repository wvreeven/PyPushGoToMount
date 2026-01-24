__all__ = ["EmulatedPlateSolver"]

import asyncio

from astropy.coordinates import SkyCoord  # type: ignore

from ..plate_solver import BasePlateSolver


class EmulatedPlateSolver(BasePlateSolver):
    async def solve(self) -> SkyCoord:
        await asyncio.sleep(0.01)
        raise RuntimeError("Exception thrown on purpose.")
