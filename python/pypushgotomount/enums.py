__all__ = [
    "AQ",
    "COLON",
    "EMPTY_REPLY",
    "HASH",
    "IDENTITY",
    "MILLISECOND",
    "TELESCOPE_REDUCTION_06INCH",
    "TELESCOPE_REDUCTION_12INCH",
    "TELESCOPE_REDUCTION_20INCH",
    "CommandName",
    "CoordinatePrecision",
    "MotorControllerState",
    "MotorControllerType",
    "SlewDirection",
    "SlewRate",
]

import enum

import numpy as np

# Algnment query.
AQ = b"\x06"

# Commands start with a colon symbol.
COLON = b":"

# Commands and some replies are terminated by the hash symbol.
HASH: bytes = b"#"

# SkySafari expects a reply to an emtpy request.
EMPTY_REPLY = "A"

# Reduction of the telescope gear additional to the motor gear reduction.
TELESCOPE_REDUCTION_06INCH = 10.82
TELESCOPE_REDUCTION_12INCH = 20.95
TELESCOPE_REDUCTION_20INCH = 34.91

# Identity (transformation) matrix.
IDENTITY = np.identity(3)

# One millisecond [sec] for asyncio waiting.
MILLISECOND = 0.001


class CommandName(enum.StrEnum):
    """LX200 Command Name."""

    CM = "CM"
    D = "D"
    G_LOWER_C = "Gc"
    G_UPPER_C = "GC"
    GD = "GD"
    G_LOWER_G = "Gg"
    G_UPPER_G = "GG"
    GL = "GL"
    GM = "GM"
    GR = "GR"
    G_LOWER_T = "Gt"
    G_UPPER_T = "GT"
    GVD = "GVD"
    GVF = "GVF"
    GVN = "GVN"
    GVP = "GVP"
    GVT = "GVT"
    GW = "GW"
    H = "H"
    Mn = "Mn"
    Me = "Me"
    M_LOWER_S = "Ms"
    Mw = "Mw"
    M_UPPER_S = "MS"
    Qn = "Qn"
    Qe = "Qe"
    Qs = "Qs"
    Qw = "Qw"
    # In general the keys should not contain the trailing '#' but in
    # this case it is necessary to avoid confusion with the other
    # commands starting with 'Q'.
    Q_HASH = "Q#"
    RC = "RC"
    RG = "RG"
    RM = "RM"
    RS = "RS"
    SC = "SC"
    Sd = "Sd"
    S_LOWER_G = "Sg"
    S_UPPER_G = "SG"
    SL = "SL"
    Sr = "Sr"
    St = "St"
    U = "U"


class CoordinatePrecision(enum.IntEnum):
    """Coordinate precision."""

    LOW = enum.auto()
    HIGH = enum.auto()


class MotorControllerState(enum.IntEnum):
    """State of a motor controller."""

    STOPPED = enum.auto()
    STOPPING = enum.auto()
    TRACKING = enum.auto()
    SLEWING = enum.auto()


class MotorControllerType(enum.IntEnum):
    """Type of motor controller."""

    NONE = enum.auto()
    CAMERA_ONLY = enum.auto()
    MOTORS_ONLY = enum.auto()
    CAMERA_AND_MOTORS = enum.auto()


class SlewDirection(enum.StrEnum):
    """Slew direction."""

    UP = "Up"
    LEFT = "Left"
    DOWN = "Down"
    RIGHT = "Right"
    NONE = "None"


class SlewRate(float, enum.Enum):
    """Slew rate [deg/sec]."""

    CENTERING = 0.01
    GUIDING = 0.1
    FIND = 0.5
    HIGH = 1.0
