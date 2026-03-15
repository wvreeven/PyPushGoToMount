import pathlib
import tempfile
from unittest import IsolatedAsyncioTestCase, mock

import pypushgotomount

tmp_dir = tempfile.TemporaryDirectory()
tmp_config_file = pathlib.Path(tmp_dir.name) / "config.json"


@mock.patch("pypushgotomount.controller.utils.CONFIG_FILE", tmp_config_file)
class TestUtils(IsolatedAsyncioTestCase):
    async def test_utils(self) -> None:
        filename = f"{tmp_dir.name}/config.json"
        with open(filename, "w") as fp:
            fp.write(
                """
{
  "camera": {
    "module": "pypushgotomount.asi",
    "class_name": "AsiCamera",
    "focal_length": 25.0
  }
}
"""
            )
        config = pypushgotomount.controller.load_config()
        assert config.camera_module_name == "pypushgotomount.asi"
        assert config.camera_class_name == "AsiCamera"
        assert config.camera_focal_length == 25.0
