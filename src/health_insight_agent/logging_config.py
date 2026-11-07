"""ロギング設定。LOG_LEVEL環境変数で変更できる(既定INFO)。"""

from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return logger
