# This code is inspired by
# https://github.com/lsst-ts/ts_genericcamera/blob/develop/python/lsst/ts/genericcamera/driver/zwocamera.py

__all__ = ["AsiCamera"]

import asyncio
import ctypes
import enum
import logging
import pathlib
import platform
import typing

import numpy as np

from ..camera import BaseCamera
from ..datetime_util import DatetimeUtil

EXTENSIONS = {"Linux": "so", "Darwin": "dylib"}

# The camera gain.
GAIN = 100
# The exposure time [µs].
EXPOSURE_TIME = 50000


class AsiBayerPattern(enum.Enum):
    """Bayer filter type."""

    ASI_BAYER_RG = 0
    ASI_BAYER_BG = 1
    ASI_BAYER_GR = 2
    ASI_BAYER_GB = 3


class AsiImgType(enum.Enum):
    """Image type."""

    ASI_IMG_RAW8 = 0
    ASI_IMG_RGB24 = 1
    ASI_IMG_RAW16 = 2
    ASI_IMG_Y8 = 3
    ASI_IMG_END = -1


class AsiGuideDirection(enum.Enum):
    """Moving direction when guiding."""

    ASI_GUIDE_NORTH = 0
    ASI_GUIDE_SOUTH = 1
    ASI_GUIDE_EAST = 2
    ASI_GUIDE_WEST = 3


class AsiFlipStatus(enum.Enum):
    """Image flip."""

    ASI_FLIP_NONE = 0
    ASI_FLIP_HORIZ = 1
    ASI_FLIP_VERT = 2
    ASI_FLIP_BOTH = 3


class AsiCameraMode(enum.Enum):
    """Camera Mode."""

    ASI_MODE_Normal = 0
    ASI_MODE_TRIG_SOFT_EDGE = 1
    ASI_MODE_TRIG_RISE_EDGE = 2
    ASI_MODE_TRIG_FALL_EDGE = 3
    ASI_MODE_TRIG_SOFT_LEVEL = 4
    ASI_MODE_TRIG_HIGH_LEVEL = 5
    ASI_MODE_TRIG_LOW_LEVEL = 6
    ASI_MODE_END = -1


class AsiErrorCode(enum.Enum):
    """Returned error code."""

    ASI_SUCCESS = 0
    ASI_ERROR_INVALID_INDEX = 1
    ASI_ERROR_INVALID_ID = 2
    ASI_ERROR_INVALID_CONTROL_TYPE = 3
    ASI_ERROR_CAMERA_CLOSED = 4
    ASI_ERROR_CAMERA_REMOVED = 5
    ASI_ERROR_INVALID_PATH = 6
    ASI_ERROR_INVALID_FILEFORMAT = 7
    ASI_ERROR_INVALID_SIZE = 8
    ASI_ERROR_INVALID_IMGTYPE = 9
    ASI_ERROR_OUTOF_BOUNDARY = 10
    ASI_ERROR_TIMEOUT = 11
    ASI_ERROR_INVALID_SEQUENCE = 12
    ASI_ERROR_BUFFER_TOO_SMALL = 13
    ASI_ERROR_VIDEO_MODE_ACTIVE = 14
    ASI_ERROR_EXPOSURE_IN_PROGRESS = 15
    ASI_ERROR_GENERAL_ERROR = 16
    ASI_ERROR_END = 17


class AsiBool(enum.Enum):
    """True or false."""

    ASI_FALSE = 0
    ASI_TRUE = 1


class AsiCameraInfoStruct(ctypes.Structure):
    """Camera information"""

    _fields_ = [
        ("Name", ctypes.c_char * 64),
        ("CameraID", ctypes.c_int),
        ("MaxHeight", ctypes.c_long),
        ("MaxWidth", ctypes.c_long),
        ("IsColorCam", ctypes.c_int),
        ("BayerPattern", ctypes.c_int),
        ("SupportedBins", ctypes.c_int * 16),
        ("SupportedVideoFormat", ctypes.c_int * 8),
        ("PixelSize", ctypes.c_double),  # micrometer
        ("MechanicalShutter", ctypes.c_int),
        ("ST4Port", ctypes.c_int),
        ("IsUSB3Host", ctypes.c_int),
        ("IsUSB3Camera", ctypes.c_int),
        ("ElecPerADU", ctypes.c_float),
        ("BitDepth", ctypes.c_int),
        ("IsTriggerCam", ctypes.c_int),
        ("Unused", ctypes.c_char * 16),
    ]


class AsiControlType(enum.Enum):
    """Camera control type."""

    ASI_GAIN = 0
    ASI_EXPOSURE = 1
    ASI_GAMMA = 2
    ASI_WB_R = 3
    ASI_WB_B = 4
    ASI_BRIGHTNESS = 5  # == OFFSET
    ASI_BANDWIDTHOVERLOAD = 6
    ASI_OVERCLOCK = 7
    ASI_TEMPERATURE = 8  # 10 * temperature
    ASI_FLIP = 9
    ASI_AUTO_MAX_GAIN = 10
    ASI_AUTO_MAX_EXP = 11  # microseconds
    ASI_AUTO_MAX_BRIGHTNESS = 12
    ASI_HARDWARE_BIN = 13
    ASI_HIGH_SPEED_MODE = 14
    ASI_COOLER_POWER_PERC = 15
    ASI_TARGET_TEMP = 16  # Not *10
    ASI_COOLER_ON = 17
    ASI_MONO_BIN = 18
    ASI_FAN_ON = 19
    ASI_PATTERN_ADJUST = 20
    ASI_ANTI_DEW_HEATER = 21


class AsiControlCapsStruct(ctypes.Structure):
    """Capacity or value ranges of control type.

    Notes
    -----
    Maximum and minimum value of ASI_TEMPERATURE is multiplied by 10.
    """

    _fields_ = [
        ("Name", ctypes.c_char * 64),
        ("Description", ctypes.c_char * 128),
        ("MaxValue", ctypes.c_long),
        ("MinValue", ctypes.c_long),
        ("DefaultValue", ctypes.c_long),
        ("IsAutoSupported", ctypes.c_int),
        ("IsWritable", ctypes.c_int),
        ("ControlType", ctypes.c_int),
        ("Unused", ctypes.c_char * 32),
    ]


class AsiExposureStatus(enum.Enum):
    """Use under snapshot mode to obtain exposure status."""

    ASI_EXP_IDLE = 0
    ASI_EXP_WORKING = 1
    ASI_EXP_SUCCESS = 2
    ASI_EXP_FAILED = 3


class AsiIdStruct(ctypes.Structure):
    """ID to be written into camera flash, 8 bytes totally."""

    _fields_ = [("id", ctypes.c_char * 8)]


class AsiSupportedModeStruct(ctypes.Structure):
    """Supported mode is used to save all supported modes returned by the camera."""

    _fields_ = [("SupportedCameraMode", ctypes.c_int * 16)]


def _get_lib_dir_and_extension() -> typing.Tuple[str, str]:
    """Use the platform module to get the lib_dir and extension.

    Linux uses the ".so" extension and macOS ".dylib". For Linux several Intel and ARM architectures are
    supported by the ASI SDK and this function attempts to support them all as well.

    Returns
    -------
    typing.Tuple[str, str]
        A tuple containing the lib_dir and extension.
    """
    pf = platform.system()
    arch = platform.machine()
    extension = EXTENSIONS[pf]
    lib_dir = ""
    match pf:
        case "Darwin":
            if arch == "arm64":
                lib_dir = "mac"
            if arch == "x86_64":
                lib_dir = "mac"
        case "Linux":
            match arch:
                case "armv6":
                    lib_dir = "armv6"
                case "armv7":
                    lib_dir = "armv7"
                case "aarch64":
                    lib_dir = "armv8"
                case "x86_64":
                    lib_dir = "x64"
                case "x86":
                    lib_dir = "x86"
    return lib_dir, extension


class AsiLib:
    def __init__(self) -> None:
        lib_dir, extension = _get_lib_dir_and_extension()
        libname = pathlib.Path(__file__).parent / "zwo" / "lib" / lib_dir / f"libASICamera2.{extension}"

        self.lib = ctypes.CDLL(str(libname))

        # LOOOONG list of argument and return types for the ASI library.
        self.lib.ASIGetNumOfConnectedCameras.restype = ctypes.c_int
        self.lib.ASIGetProductIDs.argtypes = [ctypes.c_int * 256]
        self.lib.ASIGetProductIDs.restype = ctypes.c_int
        self.lib.ASIGetCameraProperty.argtypes = [
            ctypes.POINTER(AsiCameraInfoStruct),
            ctypes.c_int,
        ]
        self.lib.ASIGetCameraProperty.restype = ctypes.c_int
        self.lib.ASIGetCameraPropertyByID.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(AsiCameraInfoStruct),
        ]
        self.lib.ASIGetCameraPropertyByID.restype = ctypes.c_int
        self.lib.ASIOpenCamera.argtypes = [ctypes.c_int]
        self.lib.ASIOpenCamera.restype = ctypes.c_int
        self.lib.ASIInitCamera.argtypes = [ctypes.c_int]
        self.lib.ASIInitCamera.restype = ctypes.c_int
        self.lib.ASICloseCamera.argtypes = [ctypes.c_int]
        self.lib.ASICloseCamera.restype = ctypes.c_int
        self.lib.ASIGetNumOfControls.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetNumOfControls.restype = ctypes.c_int
        self.lib.ASIGetControlCaps.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(AsiControlCapsStruct),
        ]
        self.lib.ASIGetControlCaps.restype = ctypes.c_int
        self.lib.ASIGetControlValue.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetControlValue.restype = ctypes.c_int
        self.lib.ASISetControlValue.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
            ctypes.c_int,
        ]
        self.lib.ASISetControlValue.restype = ctypes.c_int
        self.lib.ASISetROIFormat.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.ASISetROIFormat.restype = ctypes.c_int
        self.lib.ASIGetROIFormat.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetROIFormat.restype = ctypes.c_int
        self.lib.ASISetStartPos.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ASISetStartPos.restype = ctypes.c_int
        self.lib.ASIGetStartPos.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetStartPos.restype = ctypes.c_int
        self.lib.ASIGetDroppedFrames.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetDroppedFrames.restype = ctypes.c_int
        self.lib.ASIEnableDarkSubtract.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.ASIEnableDarkSubtract.restype = ctypes.c_int
        self.lib.ASIDisableDarkSubtract.argtypes = [ctypes.c_int]
        self.lib.ASIDisableDarkSubtract.restype = ctypes.c_int
        self.lib.ASIStartVideoCapture.argtypes = [ctypes.c_int]
        self.lib.ASIStartVideoCapture.restype = ctypes.c_int
        self.lib.ASIStopVideoCapture.argtypes = [ctypes.c_int]
        self.lib.ASIStopVideoCapture.restype = ctypes.c_int
        self.lib.ASIGetVideoData.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_int,
        ]
        self.lib.ASIGetVideoData.restype = ctypes.c_int
        self.lib.ASIPulseGuideOn.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ASIPulseGuideOn.restype = ctypes.c_int
        self.lib.ASIPulseGuideOff.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ASIPulseGuideOff.restype = ctypes.c_int
        self.lib.ASIStartExposure.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ASIStartExposure.restype = ctypes.c_int
        self.lib.ASIStopExposure.argtypes = [ctypes.c_int]
        self.lib.ASIStopExposure.restype = ctypes.c_int
        self.lib.ASIGetExpStatus.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.ASIGetExpStatus.restype = ctypes.c_int
        self.lib.ASIGetDataAfterExp.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_long,
        ]
        self.lib.ASIGetDataAfterExp.restype = ctypes.c_int
        self.lib.ASIGetID.argtypes = [ctypes.c_int, ctypes.POINTER(AsiIdStruct)]
        self.lib.ASIGetID.restype = ctypes.c_int
        self.lib.ASISetID.argtypes = [ctypes.c_int, ctypes.POINTER(AsiIdStruct)]
        self.lib.ASISetID.restype = ctypes.c_int
        self.lib.ASIGetGainOffset.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetGainOffset.restype = ctypes.c_int
        self.lib.ASIGetSDKVersion.restype = ctypes.c_char_p
        self.lib.ASIGetCameraSupportMode.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(AsiSupportedModeStruct),
        ]
        self.lib.ASIGetCameraSupportMode.restype = ctypes.c_int
        self.lib.ASIGetCameraMode.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ASIGetCameraMode.restype = ctypes.c_int
        self.lib.ASISetCameraMode.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ASISetCameraMode.restype = ctypes.c_int
        self.lib.ASISendSoftTrigger.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ASISendSoftTrigger.restype = ctypes.c_int
        self.lib.ASIGetSerialNumber.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(AsiIdStruct),
        ]
        self.lib.ASIGetSerialNumber.restype = ctypes.c_int


class AsiCamera(BaseCamera):
    def __init__(self, log: logging.Logger) -> None:
        super().__init__(log=log)

        self.asi_lib = AsiLib()
        self.asi_lib.lib.ASIGetNumOfConnectedCameras()

        self.exposure_time = 0

    async def open(self) -> None:
        self.log.debug("Opening.")
        error_code = AsiErrorCode(self.asi_lib.lib.ASIOpenCamera(self.camera_id))
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

        error_code = AsiErrorCode(self.asi_lib.lib.ASIInitCamera(self.camera_id))
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

        await self._get_image_parameters()
        await self._set_gain()
        await self._set_exposure_time()

    async def start_imaging(self) -> None:
        self.log.debug("Start imaging.")
        error_code = AsiErrorCode(self.asi_lib.lib.ASIStartVideoCapture(self.camera_id))
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

    async def stop_imaging(self) -> None:
        self.log.debug("Stop imaging.")
        error_code = AsiErrorCode(self.asi_lib.lib.ASIStopVideoCapture(self.camera_id))
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

    async def _get_image_parameters(self) -> None:
        self.log.debug("Getting image parameters.")
        camera_info_struct = AsiCameraInfoStruct()
        error_code = AsiErrorCode(
            self.asi_lib.lib.ASIGetCameraPropertyByID(self.camera_id, camera_info_struct)
        )
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

        self.img_width = camera_info_struct.MaxWidth
        self.img_height = camera_info_struct.MaxHeight
        self.pixel_size = camera_info_struct.PixelSize
        self.log.debug(f"{self.img_width=}, {self.img_height=}, {self.pixel_size=}")

        # Make sure that the camera is in NORMAL mode.
        self.log.debug("Ensuring NORMAL mode.")
        camera_support_mode_struct = AsiSupportedModeStruct()
        error_code = AsiErrorCode(
            self.asi_lib.lib.ASIGetCameraSupportMode(self.camera_id, camera_support_mode_struct)
        )
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code
        assert camera_support_mode_struct.SupportedCameraMode[0] == 0
        assert camera_support_mode_struct.SupportedCameraMode[1] == -1

        self.log.debug("Setting 16 bits.")
        error_code = AsiErrorCode(
            self.asi_lib.lib.ASISetROIFormat(
                self.camera_id,
                int(self.img_width),
                int(self.img_height),
                1,
                AsiImgType.ASI_IMG_RAW16.value,
            )
        )
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

        error_code = AsiErrorCode(self.asi_lib.lib.ASISetStartPos(self.camera_id, 0, 0))
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

    async def _set_gain(self) -> None:
        self.log.debug(f"Setting gain={GAIN}.")
        error_code = AsiErrorCode(
            self.asi_lib.lib.ASISetControlValue(
                self.camera_id,
                AsiControlType.ASI_GAIN.value,
                GAIN,
                AsiBool.ASI_FALSE.value,
            )
        )
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

    async def _set_exposure_time(self) -> None:
        self.log.debug(f"Setting exposure_time={EXPOSURE_TIME}.")
        self.exposure_time = EXPOSURE_TIME
        error_code = AsiErrorCode(
            self.asi_lib.lib.ASISetControlValue(
                self.camera_id,
                AsiControlType.ASI_EXPOSURE.value,
                self.exposure_time,
                AsiBool.ASI_FALSE.value,
            )
        )
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

    async def get_image(self) -> np.ndarray:
        self.log.debug("Get the latest image.")
        return await asyncio.to_thread(self._blocking_get_image)

    def _blocking_get_image(self) -> np.ndarray:
        img_start = DatetimeUtil.get_timestamp()
        img_buffer_size = self.img_width * self.img_height * 2  # 16 bit data == 2 bytes
        img_buffer = ctypes.create_string_buffer(img_buffer_size)
        self.log.debug("Blocking get the latest image.")
        error_code = AsiErrorCode(
            self.asi_lib.lib.ASIGetVideoData(self.camera_id, img_buffer, img_buffer_size, self.exposure_time)
        )
        assert error_code == AsiErrorCode.ASI_SUCCESS, error_code

        data_array = np.frombuffer(img_buffer, dtype=np.uint16)
        data = data_array.reshape(self.img_height, self.img_width)
        img_end = DatetimeUtil.get_timestamp()
        self.log.debug(f"Getting latest image took {img_end - img_start} s.")
        return data
