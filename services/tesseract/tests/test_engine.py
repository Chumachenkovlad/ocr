"""Tests for the engine module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.engine import (
    TesseractError,
    TesseractNotFoundError,
    _check_tesseract_available,
    run_tesseract,
)


class TestRunTesseract:
    """Tests for run_tesseract()."""

    def test_success(self, tmp_path: Path) -> None:
        """Successful Tesseract invocation returns hOCR output."""
        fake_hocr = "<html><body>fake hocr</body></html>"
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake image data")

        with patch("src.engine.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=fake_hocr, stderr=""
            )
            result = run_tesseract(image_file, lang="eng", dpi=300, timeout=30)

        assert result == fake_hocr
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "tesseract" in call_args[0][0][0]
        assert "--dpi" in call_args[0][0]
        assert "hocr" in call_args[0][0]

    def test_binary_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError raises TesseractNotFoundError."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        with patch("src.engine.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(TesseractNotFoundError, match="not found"):
                run_tesseract(image_file, binary="/nonexistent/tesseract")

    def test_timeout(self, tmp_path: Path) -> None:
        """TimeoutExpired raises TesseractError."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        with patch(
            "src.engine.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="tesseract", timeout=5),
        ):
            with pytest.raises(TesseractError, match="timed out"):
                run_tesseract(image_file, timeout=5)

    def test_nonzero_exit(self, tmp_path: Path) -> None:
        """Non-zero exit code raises TesseractError."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        with patch("src.engine.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Error opening data file"
            )
            with pytest.raises(TesseractError, match="OCR failed"):
                run_tesseract(image_file)

    def test_stderr_not_found_message(self, tmp_path: Path) -> None:
        """stderr containing 'not found' raises TesseractNotFoundError."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"data")

        with patch("src.engine.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="tesseract not found"
            )
            with pytest.raises(TesseractNotFoundError, match="not found"):
                run_tesseract(image_file)


class TestCheckTesseractAvailable:
    """Tests for _check_tesseract_available()."""

    def test_available(self) -> None:
        with patch("src.engine.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="tesseract 5.3.0", stderr=""
            )
            assert _check_tesseract_available("tesseract") is True

    def test_not_available(self) -> None:
        with patch("src.engine.subprocess.run", side_effect=FileNotFoundError):
            assert _check_tesseract_available("tesseract") is False

    def test_timeout(self) -> None:
        with patch(
            "src.engine.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="tesseract", timeout=5),
        ):
            assert _check_tesseract_available("tesseract") is False
