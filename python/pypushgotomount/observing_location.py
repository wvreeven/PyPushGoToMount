__all__ = ["get_observing_location", "set_height", "set_latitude", "set_longitude"]

from astropy import units as u
from astropy.coordinates import EarthLocation, Latitude, Longitude

_observing_location: EarthLocation = EarthLocation.from_geodetic(
    lon=Longitude("0d0m0s"),
    lat=Latitude("0d0m00s"),
    height=0 * u.meter,
)


def set_longitude(longitude: Longitude) -> None:
    """
    Set the longitude of the observing_location. It will create a new
    `astropy.coordinates.EarthLocation` instance with the new value of longitude
    while copying the other values.

    Parameters
    ----------
    longitude: `Longitude`
        The new longitude
    """
    global _observing_location
    _observing_location = EarthLocation.from_geodetic(
        lon=longitude,
        lat=_observing_location.lat,
        height=_observing_location.height,
    )


def set_latitude(latitude: Latitude) -> None:
    """
    Set the latitude of the observing_location. It will create a new
    `astropy.coordinates.EarthLocation` instance with the new value of latitude
    while copying the other values.

    Parameters
    ----------
    latitude: `Latitude`
        The new latitude
    """
    global _observing_location
    _observing_location = EarthLocation.from_geodetic(
        lon=_observing_location.lon,
        lat=latitude,
        height=_observing_location.height,
    )


def set_height(height: float) -> None:
    """
    Set the height of the observing_location. It will create a new
    `astropy.coordinates.EarthLocation` instance with the new value of latitude
    while copying the other values.

    Parameters
    ----------
    height: `float`
        The new height in `u.meter`
    """
    global _observing_location
    _observing_location = EarthLocation.from_geodetic(
        lon=_observing_location.lon,
        lat=_observing_location.lat,
        height=height * u.meter,
    )


def get_observing_location() -> EarthLocation:
    return _observing_location
