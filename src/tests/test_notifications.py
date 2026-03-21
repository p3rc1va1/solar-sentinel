"""Tests for app.services.notifications."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.notifications import NotificationService


@pytest.fixture
def email_service():
    return NotificationService(
        email_enabled=True,
        email_address="test@example.com",
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_username="user@test.com",
        smtp_password="pass123",
    )


@pytest.fixture
def telegram_service():
    return NotificationService(
        telegram_enabled=True,
        telegram_bot_token="123:ABC",
        telegram_chat_id="456",
    )


class TestNotificationService:
    def test_defaults(self):
        svc = NotificationService()
        assert svc.email_enabled is False
        assert svc.telegram_enabled is False

    def test_update_settings(self):
        svc = NotificationService()
        svc.update_settings(email_enabled=True, email_address="new@test.com")
        assert svc.email_enabled is True
        assert svc.email_address == "new@test.com"

    def test_update_settings_ignores_unknown(self):
        svc = NotificationService()
        svc.update_settings(unknown_field="value")
        assert not hasattr(svc, "unknown_field")

    def test_bot_cache_invalidated_on_token_change(self, telegram_service):
        telegram_service._bot = MagicMock()
        telegram_service.update_settings(telegram_bot_token="new-token")
        assert telegram_service._bot is None

    @pytest.mark.asyncio
    async def test_send_report_no_channels(self):
        svc = NotificationService()
        result = await svc.send_report("test", "WARNING")
        assert result == {}

    @pytest.mark.asyncio
    @patch("app.services.notifications.aiosmtplib.send", new_callable=AsyncMock)
    async def test_send_email_success(self, mock_send, email_service):
        result = await email_service.send_report("# Report", "CRITICAL")
        assert result["email"] is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.notifications.aiosmtplib.send", new_callable=AsyncMock, side_effect=Exception("SMTP error"))
    async def test_send_email_failure(self, mock_send, email_service):
        result = await email_service.send_report("# Report", "CRITICAL")
        assert result["email"] is False

    @pytest.mark.asyncio
    @patch("app.services.notifications.Bot")
    async def test_send_telegram_success(self, mock_bot_cls, telegram_service):
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.token = "123:ABC"
        mock_bot_cls.return_value = mock_bot

        result = await telegram_service.send_report("# Report", "WARNING")
        assert result["telegram"] is True

    @pytest.mark.asyncio
    @patch("app.services.notifications.Bot")
    async def test_send_telegram_failure(self, mock_bot_cls, telegram_service):
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("API error"))
        mock_bot.token = "123:ABC"
        mock_bot_cls.return_value = mock_bot

        result = await telegram_service.send_report("# Report", "CRITICAL")
        assert result["telegram"] is False

    @pytest.mark.asyncio
    @patch("app.services.notifications.aiosmtplib.send", new_callable=AsyncMock)
    async def test_send_email_with_image(self, mock_send, email_service, tmp_path):
        from PIL import Image
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (10, 10)).save(str(img_path), "JPEG")
        result = await email_service.send_report("# Report", "WARNING", str(img_path))
        assert result["email"] is True
