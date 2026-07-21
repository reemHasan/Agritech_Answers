# ---------------------------------------------------------------------------
# Structured (JSON) logging
# ---------------------------------------------------------------------------
# One JSON object per log line, written to stdout -- the standard contract
# for containerized services, so a log aggregator (ELK, Loki, CloudWatch,
# etc.) can parse and index fields directly instead of grepping free text.

# Attribute names LogRecord already carries by default -- anything else on
# the record came from an `extra={...}` kwarg passed to a logging call, and
# gets folded into the JSON output as its own field (e.g. request_id,
# duration_ms, crop).
from datetime import datetime, timezone
import json
import logging
import os

_RESERVED_LOG_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Emit every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_LOG_ATTRS}
        log_obj.update(extras)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))