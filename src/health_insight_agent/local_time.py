"""ユーザーの生活時間帯(日本時間)に関するヘルパー。

既存アプリ(Python-Smart-Health-Intelligence-Platform)の local_time.py と同じ方針。
DBの日時はUTC(timestamptz)で持つが、「何日の記録か」はJSTで区切る。
UTCのまま`.date()`を取ると、JST 0:00〜8:59の記録(朝食・起床)が前日扱いになるため。
睡眠は「起床時刻の日付」に属する(既存アプリと同じ)。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

APP_TIMEZONE = ZoneInfo("Asia/Tokyo")


def local_date_of(dt: datetime) -> date:
    """aware datetimeをJSTの日付に変換する。"""
    return dt.astimezone(APP_TIMEZONE).date()


def local_today() -> date:
    """JSTの今日(実行環境のTZに依存しない)。"""
    return local_date_of(datetime.now(UTC))


def local_datetime(d: date, t: time) -> datetime:
    """JSTの日付+時刻からaware datetimeを作る。"""
    return datetime.combine(d, t, tzinfo=APP_TIMEZONE)


def local_day_range(start: date, end: date) -> tuple[datetime, datetime]:
    """JSTの[start, end]を含む期間を、半開区間[開始, 終了)のaware datetimeにする。

    Repositoryは半開区間で問い合わせる(境界の二重カウントを避けるため)。
    """
    if end < start:
        raise ValueError("end must be >= start")
    return (
        local_datetime(start, time.min),
        local_datetime(end + timedelta(days=1), time.min),
    )
