from __future__ import annotations

import pytest

import memnet_agent.sleeping as sleeping
from memnet_agent import SleepConfig


def test_default_utc_does_not_require_zoneinfo_database(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_zoneinfo(name: str):
        raise sleeping.ZoneInfoNotFoundError(name)

    monkeypatch.setattr(sleeping, "ZoneInfo", missing_zoneinfo)

    config = SleepConfig.idle()
    assert config.timezone == "UTC"


def test_missing_non_utc_timezone_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_zoneinfo(name: str):
        raise sleeping.ZoneInfoNotFoundError(name)

    monkeypatch.setattr(sleeping, "ZoneInfo", missing_zoneinfo)

    with pytest.raises(ValueError, match="Install the 'tzdata' package"):
        SleepConfig.scheduled(timezone="Europe/Moscow")
