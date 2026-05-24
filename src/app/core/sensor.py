"""DHT22 (AM2302) temperature and humidity sensor interface.

Uses adafruit_dht on Raspberry Pi, falls back to a stub on other platforms
for development purposes.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import board  # type: ignore[import-untyped]
    import adafruit_dht  # type: ignore[import-untyped]
    _HAS_DHT = True
except ImportError:
    _HAS_DHT = False


class DHTSensor:
    """Wrapper around Adafruit DHT22 sensor with development fallback."""

    def __init__(self, pin=None) -> None:
        self._sensor = None
        self._pin = pin

    def start(self) -> None:
        """Initialize the sensor."""
        if _HAS_DHT:
            try:
                gpio_pin = self._pin or board.D4  # Default: GPIO4
                self._sensor = adafruit_dht.DHT22(gpio_pin)
                logger.info("DHT22 sensor started on pin %s", gpio_pin)
            except Exception as e:
                logger.warning("DHT22 init failed: %s — running in stub mode", e)
                self._sensor = None
        else:
            logger.info("DHT22 running in stub mode (no adafruit_dht)")

    def stop(self) -> None:
        """Release sensor resources."""
        if self._sensor is not None:
            try:
                self._sensor.exit()
            except Exception:
                pass
            self._sensor = None
            logger.info("DHT22 sensor stopped")

    def read(self) -> dict | None:
        """Read current temperature and humidity.

        Returns:
            dict with 'temperature' (°C) and 'humidity' (%), or None if unavailable.
        """
        if self._sensor is None:
            return None

        try:
            temp = self._sensor.temperature
            hum = self._sensor.humidity
            if temp is not None and hum is not None:
                return {"temperature": float(temp), "humidity": float(hum)}
        except RuntimeError as e:
            # DHT sensors occasionally fail reads — this is normal
            logger.debug("DHT22 read error (transient): %s", e)
        except Exception as e:
            logger.warning("DHT22 read error: %s", e)

        return None

    @property
    def is_available(self) -> bool:
        """Check if a real sensor is connected."""
        return self._sensor is not None
