"""Notification service — email and Telegram delivery."""

import logging
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import aiosmtplib
from telegram import Bot

logger = logging.getLogger(__name__)


class NotificationService:
    """Sends reports via user-configured channels (email + Telegram)."""

    def __init__(
        self,
        email_enabled: bool = False,
        email_address: str = "",
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_username: str = "",
        smtp_password: str = "",
        telegram_enabled: bool = False,
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
    ) -> None:
        self.email_enabled = email_enabled
        self.email_address = email_address
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.telegram_enabled = telegram_enabled
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self._bot: Bot | None = None

    def update_settings(self, **kwargs) -> None:
        """Update notification settings at runtime."""
        token_changed = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == "telegram_bot_token" and value != self.telegram_bot_token:
                    token_changed = True
                setattr(self, key, value)
        if token_changed:
            self._bot = None  # force re-creation

    def _get_bot(self) -> Bot:
        """Get or create cached Telegram Bot instance."""
        if self._bot is None or self._bot.token != self.telegram_bot_token:
            self._bot = Bot(token=self.telegram_bot_token)
        return self._bot

    async def send_report(
        self,
        report_markdown: str,
        severity: str,
        image_path: str | None = None,
    ) -> dict[str, bool]:
        """Send report to all enabled channels. Returns {channel: success}."""
        results = {}

        if self.email_enabled and self.email_address:
            results["email"] = await self._send_email(report_markdown, severity, image_path)

        if self.telegram_enabled and self.telegram_bot_token and self.telegram_chat_id:
            results["telegram"] = await self._send_telegram(report_markdown, severity, image_path)

        if not results:
            logger.warning("No notification channels enabled")

        return results

    async def _send_email(
        self, report_markdown: str, severity: str, image_path: str | None
    ) -> bool:
        """Send report via email."""
        try:
            msg = MIMEMultipart("mixed")
            msg["From"] = self.smtp_username
            msg["To"] = self.email_address
            msg["Subject"] = f"[Solar Sentinel] {severity} — Panel Defect Report"

            color = {"CRITICAL": "#dc3545", "WARNING": "#ffc107"}.get(severity, "#28a745")
            html = (
                f'<html><body style="font-family: Arial, sans-serif; padding: 20px;">'
                f'<h2 style="color: {color};">Solar Panel {severity} Alert</h2>'
                f'<pre style="white-space: pre-wrap; font-family: monospace; '
                f'background: #f8f9fa; padding: 16px; border-radius: 8px;">'
                f'{report_markdown}</pre><hr>'
                f'<p style="color: #6c757d; font-size: 12px;">'
                f'Sent by Solar Sentinel — Autonomous Solar Panel Monitoring</p>'
                f'</body></html>'
            )
            msg.attach(MIMEText(html, "html"))

            if image_path and Path(image_path).exists():
                with open(image_path, "rb") as f:
                    msg.attach(MIMEImage(f.read(), name=Path(image_path).name))

            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=False,
                start_tls=True,
            )
            logger.info("Email sent to %s", self.email_address)
            return True

        except Exception as e:
            logger.error("Email send failed: %s", e)
            return False

    async def _send_telegram(
        self, report_markdown: str, severity: str, image_path: str | None
    ) -> bool:
        """Send report via Telegram bot."""
        try:
            bot = self._get_bot()
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(severity, "⚪")
            text = f"{icon} *Solar Sentinel — {severity}*\n\n{report_markdown}"
            if len(text) > 4000:
                text = text[:4000] + "\n\n_(truncated)_"

            await bot.send_message(
                chat_id=self.telegram_chat_id,
                text=text,
                parse_mode="Markdown",
            )

            if image_path and Path(image_path).exists():
                with open(image_path, "rb") as photo:
                    await bot.send_photo(
                        chat_id=self.telegram_chat_id,
                        photo=photo,
                        caption=f"{severity} detection — see report above",
                    )

            logger.info("Telegram message sent to chat %s", self.telegram_chat_id)
            return True

        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False
