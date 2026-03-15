import asyncio
import logging
import unittest

import pypushgotomount


class TestPushGoToMount(unittest.IsolatedAsyncioTestCase):
    async def test_pushgoto_mount(self) -> None:
        self.log = logging.getLogger(type(self).__name__)
        async with pypushgotomount.PushGoToMount(run_forever=False) as self.lx200_mount:
            reader, writer = await asyncio.open_connection(host="localhost", port=11880)
            writer.write(b"#")
            await writer.drain()
            # No reply expected.

            writer.write(b"\x06")
            await writer.drain()
            data = await reader.read(1)
            assert data == b"A"

            writer.write(b":H#")
            await writer.drain()
            # No reply expected.

            writer.write(b"!")
            await writer.drain()
            # No reply expected.

            writer.write(b":bla#")
            await writer.drain()
            # No reply expected.

            writer.write(b":SG+01.0#")
            await writer.drain()
            data = await reader.read(1)
            assert data == b"1"

            writer.write(b":SL00:00:00#")
            await writer.drain()
            data = await reader.read(1)
            assert data == b"1"

            writer.write(b":SC03/14/25#")
            await writer.drain()
            data = await reader.read(1)
            assert data == b"1"
            data = await reader.readuntil(b"#")
            assert data == pypushgotomount.controller.UPDATING_PLANETARY_DATA1.encode("utf8")
            data = await reader.readuntil(b"#")
            assert data == pypushgotomount.controller.UPDATING_PLANETARY_DATA2.encode("utf8")

            writer.write(b":GR#")
            await writer.drain()
            data = await reader.readuntil(b"#")
            assert data.decode().endswith("#")

            await self.lx200_mount.stop()
