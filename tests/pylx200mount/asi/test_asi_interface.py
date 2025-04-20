import ctypes
import logging
from unittest import IsolatedAsyncioTestCase, mock

import pylx200mount


class TestAsiInterface(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with mock.patch("pylx200mount.asi.asi_interface.ctypes.CDLL") as mock_cdll:
            mock_asi_lib = mock.MagicMock()
            mock_cdll.return_value = mock_asi_lib
            mock_asi_lib.ASIOpenCamera.return_value = 0
            mock_asi_lib.ASIInitCamera.return_value = 0
            mock_asi_lib.ASIGetCameraPropertyByID.return_value = 0
            mock_asi_lib.ASIGetCameraSupportMode.side_effect = self.set_asi_supported_mode_struct
            mock_asi_lib.ASISetROIFormat.return_value = 0
            mock_asi_lib.ASISetStartPos.return_value = 0
            mock_asi_lib.ASISetControlValue.return_value = 0
            mock_asi_lib.ASIStartVideoCapture.return_value = 0
            mock_asi_lib.ASIStopVideoCapture.return_value = 0
            mock_asi_lib.ASIGetVideoData.return_value = 0
            self.log = logging.getLogger(type(self).__name__)
            self.asi_camera = pylx200mount.asi.AsiCamera(log=self.log)

    async def test_open(self) -> None:
        await self.asi_camera.open()

    async def test_start_imaging(self) -> None:
        await self.asi_camera.start_imaging()

    async def test_stop_imaging(self) -> None:
        await self.asi_camera.stop_imaging()

    async def test_get_image(self) -> None:
        await self.asi_camera.get_image()

    def set_asi_supported_mode_struct(self, _: int, camera_support_mode_struct: ctypes.Structure) -> int:
        camera_support_mode_struct.SupportedCameraMode = (ctypes.c_int * 16)(
            0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        )
        return 0
