from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as clock_time, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SleepMode = Literal["off", "idle", "scheduled", "workers"]


@dataclass(frozen=True, slots=True)
class SleepConfig:
    """Controls graph consolidation outside normal inference.

    ``idle``
        Consolidate after the agent has been inactive for a configured period.
    ``scheduled``
        Disable normal inference inside a daily time window and consolidate the
        graph periodically. ``learn()`` remains available.
    ``workers``
        Run one or more daemon memory workers that continuously consolidate the
        graph while the primary agent remains available.
    ``off``
        Background maintenance is disabled; call ``agent.sleep()`` manually.
    """

    mode: SleepMode = "idle"
    idle_after_seconds: float = 30.0
    check_interval_seconds: float = 5.0
    maintenance_interval_seconds: float = 30.0
    max_syntheses: int = 3
    schedule_start: str = "02:00"
    schedule_end: str = "05:00"
    timezone: str = "UTC"
    worker_count: int = 2
    allow_interrupt: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"off", "idle", "scheduled", "workers"}:
            raise ValueError(f"Unsupported sleep mode: {self.mode}")
        if self.idle_after_seconds < 0:
            raise ValueError("idle_after_seconds must be non-negative")
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")
        if self.maintenance_interval_seconds <= 0:
            raise ValueError("maintenance_interval_seconds must be positive")
        if self.max_syntheses < 0:
            raise ValueError("max_syntheses must be non-negative")
        if self.worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        _parse_clock(self.schedule_start)
        _parse_clock(self.schedule_end)
        _resolve_timezone(self.timezone)

    @classmethod
    def off(cls) -> "SleepConfig":
        return cls(mode="off")

    @classmethod
    def idle(
        cls,
        *,
        after_seconds: float = 30.0,
        check_every_seconds: float = 5.0,
        maintenance_every_seconds: float = 30.0,
        max_syntheses: int = 3,
    ) -> "SleepConfig":
        return cls(
            mode="idle",
            idle_after_seconds=after_seconds,
            check_interval_seconds=check_every_seconds,
            maintenance_interval_seconds=maintenance_every_seconds,
            max_syntheses=max_syntheses,
        )

    @classmethod
    def scheduled(
        cls,
        *,
        start: str = "02:00",
        end: str = "05:00",
        timezone: str = "UTC",
        maintenance_every_seconds: float = 60.0,
        max_syntheses: int = 3,
        allow_interrupt: bool = False,
    ) -> "SleepConfig":
        return cls(
            mode="scheduled",
            schedule_start=start,
            schedule_end=end,
            timezone=timezone,
            maintenance_interval_seconds=maintenance_every_seconds,
            max_syntheses=max_syntheses,
            allow_interrupt=allow_interrupt,
        )

    @classmethod
    def workers(
        cls,
        *,
        count: int = 2,
        maintenance_every_seconds: float = 30.0,
        max_syntheses: int = 3,
    ) -> "SleepConfig":
        return cls(
            mode="workers",
            worker_count=count,
            maintenance_interval_seconds=maintenance_every_seconds,
            max_syntheses=max_syntheses,
        )

    def in_scheduled_window(self, moment: datetime | None = None) -> bool:
        if self.mode != "scheduled":
            return False
        zone = _resolve_timezone(self.timezone)
        now = moment.astimezone(zone) if moment is not None else datetime.now(zone)
        current = now.timetz().replace(tzinfo=None)
        start = _parse_clock(self.schedule_start)
        end = _parse_clock(self.schedule_end)
        if start == end:
            return True
        if start < end:
            return start <= current < end
        return current >= start or current < end


def _resolve_timezone(value: str):
    normalized = value.strip().upper()
    if normalized in {"UTC", "ETC/UTC", "GMT", "Z"}:
        return timezone.utc
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown timezone {value!r}. Install the 'tzdata' package or use 'UTC'."
        ) from exc


def _parse_clock(value: str) -> clock_time:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        return clock_time(hour=hour, minute=minute)
    except (ValueError, TypeError) as exc:
        raise ValueError("Time must use HH:MM format, for example '02:30'.") from exc
