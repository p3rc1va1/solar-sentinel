"""Tests for app.core.camera."""

import asyncio
from pathlib import Path

import pytest

from app.core.camera import Camera


@pytest.fixture
def cam():
    return Camera(resolution=(64, 64))


class TestCamera:
    def test_capture_frame_stub(self, cam):
        img = cam.capture_frame()
        assert img.size == (64, 64)
        assert img.mode == "RGB"

    def test_capture_to_file(self, cam, tmp_path):
        path = cam.capture_to_file(tmp_path / "test.jpg")
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_capture_to_file_creates_dirs(self, cam, tmp_path):
        path = cam.capture_to_file(tmp_path / "sub" / "dir" / "img.jpg")
        assert path.exists()

    def test_capture_jpeg_bytes(self, cam):
        data = cam.capture_jpeg_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes

    @pytest.mark.asyncio
    async def test_start_stop_stub(self, cam):
        await cam.start()
        assert cam._camera is None  # stub mode
        await cam.stop()

    @pytest.mark.asyncio
    async def test_generate_mjpeg_frames(self, cam):
        gen = cam.generate_mjpeg_frames()
        frame = await gen.__anext__()
        assert b"--frame\r\n" in frame
        assert b"Content-Type: image/jpeg" in frame
