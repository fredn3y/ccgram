"""Tests for file_handler helper functions."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.codex_history_sync import PendingCodexAction
from ccgram.handlers.file_handler import (
    _generate_photo_filename,
    _sanitize_caption,
    _sanitize_filename,
    _upload_and_notify,
    _unique_dest,
    _validate_dest_path,
)


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("document.pdf", "document.pdf"),
            ("file-name_123.txt", "file-name_123.txt"),
            ("/etc/passwd", "passwd"),
            ("../../../etc/passwd", "passwd"),
            ("../../etc/passwd", "passwd"),
            ("hello world!.txt", "hello_world_.txt"),
            ("file@#$.txt", "file___.txt"),
            ("..", "unnamed"),
            (".", "unnamed"),
            ("...", "unnamed"),
            ("", "unnamed"),
        ],
    )
    def test_sanitize(self, input_name: str, expected: str) -> None:
        assert _sanitize_filename(input_name) == expected

    def test_truncates_long_names_preserving_extension(self) -> None:
        long = "a" * 250 + ".pdf"
        result = _sanitize_filename(long)
        assert len(result) <= 200
        assert result.endswith(".pdf")


class TestUniqueDest:
    def test_returns_original_if_not_exists(self, tmp_path: Path) -> None:
        assert _unique_dest(tmp_path / "file.txt") == tmp_path / "file.txt"

    @pytest.mark.parametrize(
        ("existing_files", "expected_name"),
        [
            (["file.txt"], "file_1.txt"),
            (["file.txt", "file_1.txt", "file_2.txt"], "file_3.txt"),
            (["file"], "file_1"),
        ],
    )
    def test_increments_suffix(
        self, tmp_path: Path, existing_files: list[str], expected_name: str
    ) -> None:
        for name in existing_files:
            (tmp_path / name).write_text("x")
        assert _unique_dest(tmp_path / existing_files[0]) == tmp_path / expected_name

    def test_fallback_to_timestamp_after_100(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.txt"
        for i in range(100):
            name = "file.txt" if i == 0 else f"file_{i}.txt"
            (tmp_path / name).write_text(str(i))
        result = _unique_dest(dest)
        assert result.name.startswith("file_") and result.name.endswith(".txt")
        assert result != dest

    def test_broken_symlink_treated_as_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "file.txt"
        dest.symlink_to(tmp_path / "nonexistent_target")
        assert _unique_dest(dest) == tmp_path / "file_1.txt"


class TestValidateDestPath:
    @pytest.mark.parametrize(
        ("rel_dest", "expected"),
        [
            ("file.txt", True),
            ("subdir/file.txt", True),
            ("../outside.txt", False),
        ],
    )
    def test_path_validation(
        self, tmp_path: Path, rel_dest: str, expected: bool
    ) -> None:
        upload = tmp_path / "upload"
        upload.mkdir()
        if "/" in rel_dest and not rel_dest.startswith(".."):
            (upload / Path(rel_dest).parent).mkdir(parents=True, exist_ok=True)
        assert _validate_dest_path(upload / rel_dest, upload) is expected

    def test_rejects_absolute_path_outside(self, tmp_path: Path) -> None:
        upload = tmp_path / "upload"
        upload.mkdir()
        assert _validate_dest_path(tmp_path / "outside.txt", upload) is False


class TestSanitizeCaption:
    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            ("", ""),
            ("hello\x00\x01\x02world", "helloworld"),
            ("hello\x07\x1bworld", "helloworld"),
            ("line1\nline2\r\nline3\ttab", "line1 line2  line3\ttab"),
        ],
    )
    def test_sanitize(self, input_text: str, expected: str) -> None:
        assert _sanitize_caption(input_text) == expected

    def test_limits_to_500_chars(self) -> None:
        assert len(_sanitize_caption("a" * 600)) == 500


class TestGeneratePhotoFilename:
    def test_format(self) -> None:
        result = _generate_photo_filename("ABCDEFGHIJKLMNOP")
        assert re.match(r"^photo_\d{8}_\d{6}_ABCDEFGH\.jpg$", result)


class TestUploadAndNotify:
    def _message(self) -> MagicMock:
        message = MagicMock()
        message.caption = ""
        message.chat.id = -100999
        message.chat.send_action = AsyncMock()
        message.message_id = 500
        bot = MagicMock()
        bot.get_file = AsyncMock()
        message.get_bot.return_value = bot
        return message

    async def test_app_server_synced_photo_upload_submits_local_image(
        self, tmp_path: Path
    ) -> None:
        message = self._message()
        message.caption = "check this"

        async def download_to_drive(path: str) -> None:
            Path(path).write_bytes(b"fake image")

        file_obj = MagicMock()
        file_obj.download_to_drive = AsyncMock(side_effect=download_to_drive)
        message.get_bot.return_value.get_file.return_value = file_obj

        with (
            patch(
                "ccgram.handlers.file_handler.resolve_pending_codex_attachment_target",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    cwd=str(tmp_path),
                    app_server_thread_id="thread-1",
                    session_id="sess-1",
                ),
            ),
            patch(
                "ccgram.handlers.file_handler.submit_attachment_to_pending_codex_topic",
                new_callable=AsyncMock,
                return_value=PendingCodexAction("submitted", message="turn-1"),
            ) as mock_submit,
            patch(
                "ccgram.handlers.file_handler.ack_reaction",
                new_callable=AsyncMock,
            ) as mock_ack,
            patch(
                "ccgram.handlers.file_handler.safe_reply",
                new_callable=AsyncMock,
            ) as mock_reply,
            patch(
                "ccgram.handlers.file_handler.send_to_window",
                new_callable=AsyncMock,
            ) as mock_send_to_window,
        ):
            await _upload_and_notify(
                message,
                100,
                77,
                "shot.jpg",
                "file-id",
                123,
                "Photo",
                "I've uploaded an image to {path} — please take a look.",
                "\U0001f4f7",
                app_server_input_kind="image",
            )

        saved_path = tmp_path / ".ccgram-uploads" / "shot.jpg"
        assert saved_path.read_bytes() == b"fake image"
        mock_submit.assert_awaited_once_with(
            100,
            77,
            -100999,
            "I've uploaded an image to .ccgram-uploads/shot.jpg — please take a look.\n\n"
            "User note: check this",
            extra_input=[{"type": "localImage", "path": str(saved_path)}],
        )
        mock_send_to_window.assert_not_awaited()
        mock_ack.assert_awaited_once_with(message.get_bot.return_value, -100999, 500)
        assert "Uploaded `.ccgram-uploads/shot.jpg`" in mock_reply.call_args.args[1]

    async def test_tmux_upload_path_still_notifies_bound_window(
        self, tmp_path: Path
    ) -> None:
        message = self._message()

        async def download_to_drive(path: str) -> None:
            Path(path).write_bytes(b"report")

        file_obj = MagicMock()
        file_obj.download_to_drive = AsyncMock(side_effect=download_to_drive)
        message.get_bot.return_value.get_file.return_value = file_obj

        with (
            patch(
                "ccgram.handlers.file_handler.resolve_pending_codex_attachment_target",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("ccgram.handlers.file_handler.thread_router") as mock_tr,
            patch("ccgram.handlers.file_handler.view_window") as mock_view_window,
            patch(
                "ccgram.handlers.file_handler.send_to_window",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as mock_send_to_window,
            patch(
                "ccgram.handlers.file_handler.submit_attachment_to_pending_codex_topic",
                new_callable=AsyncMock,
            ) as mock_submit,
            patch(
                "ccgram.handlers.file_handler.ack_reaction",
                new_callable=AsyncMock,
            ),
            patch(
                "ccgram.handlers.file_handler.safe_reply",
                new_callable=AsyncMock,
            ),
        ):
            mock_tr.resolve_window_for_thread.return_value = "@11"
            mock_view_window.return_value = SimpleNamespace(cwd=str(tmp_path))

            await _upload_and_notify(
                message,
                100,
                77,
                "report.pdf",
                "file-id",
                123,
                "File",
                "I've uploaded {name} to {path}",
                "\U0001f4ce",
                app_server_input_kind="file",
            )

        mock_send_to_window.assert_awaited_once_with(
            "@11",
            "I've uploaded report.pdf to .ccgram-uploads/report.pdf",
        )
        mock_submit.assert_not_awaited()
