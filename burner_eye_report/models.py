from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_UNIT = "g/h"


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    row_number: int | None = None
    blocking: bool = False

    @property
    def severity_label(self) -> str:
        return {
            "error": "Ошибка",
            "warning": "Предупреждение",
            "info": "Информация",
        }.get(self.severity, self.severity)


@dataclass(slots=True)
class PredictionRow:
    timestamp: datetime
    frame_id: str
    source_unit: str
    expected_fuel: float
    expected_steam: float
    predicted_fuel: float
    predicted_steam: float
    expected_fuel_g_h: float
    expected_steam_g_h: float
    predicted_fuel_g_h: float
    predicted_steam_g_h: float
    trained_regime: bool
    predicted_regime: str
    regime_confidence: float | None
    source_row_number: int
    image_path: Path | None = None
    temporal_frame_count: int = 0
    regime_id: str = ""


@dataclass(slots=True)
class Regime:
    regime_id: str
    fuel_g_h: float
    steam_g_h: float
    rows: list[PredictionRow] = field(default_factory=list)
    final_status: bool | None = None

    @property
    def statuses(self) -> set[bool]:
        return {row.trained_regime for row in self.rows}

    @property
    def is_conflict(self) -> bool:
        return len(self.statuses) > 1

    @property
    def source_status(self) -> bool | None:
        statuses = self.statuses
        return next(iter(statuses)) if len(statuses) == 1 else None

    @property
    def source_status_label(self) -> str:
        if self.is_conflict:
            ones = sum(row.trained_regime for row in self.rows)
            zeros = len(self.rows) - ones
            return f"Конфликт: 1 — {ones}, 0 — {zeros}"
        return "Был в обучении" if self.source_status else "Не был в обучении"

    @property
    def final_status_label(self) -> str:
        if self.final_status is None:
            return "Требуется выбор"
        return "Был в обучении" if self.final_status else "Не был в обучении"

    @property
    def manually_changed(self) -> bool:
        return self.final_status is not None and (
            self.is_conflict or self.final_status != self.source_status
        )

    @property
    def first_timestamp(self) -> datetime:
        return min(row.timestamp for row in self.rows)

    @property
    def last_timestamp(self) -> datetime:
        return max(row.timestamp for row in self.rows)

    @property
    def source_units(self) -> list[str]:
        return sorted({row.source_unit for row in self.rows})

    @property
    def display_name(self) -> str:
        return (
            f"Пар: {format_flow(self.steam_g_h)} г/ч; "
            f"топливо: {format_flow(self.fuel_g_h)} г/ч"
        )


@dataclass(slots=True)
class ExperimentData:
    root: Path
    rows: list[PredictionRow] = field(default_factory=list)
    regimes: list[Regime] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    total_csv_rows: int = 0
    excluded_rows: int = 0
    orphan_frames: list[Path] = field(default_factory=list)
    temporal_window_count: int = 0

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.blocking]

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def primary_frame_count(self) -> int:
        return sum(row.image_path is not None for row in self.rows)

    @property
    def missing_frame_count(self) -> int:
        return len(self.rows) - self.primary_frame_count

    @property
    def conflict_count(self) -> int:
        return sum(regime.is_conflict and regime.final_status is None for regime in self.regimes)

    @property
    def trained_regime_count(self) -> int:
        return sum(regime.final_status is True for regime in self.regimes)

    @property
    def untrained_regime_count(self) -> int:
        return sum(regime.final_status is False for regime in self.regimes)

    @property
    def can_generate(self) -> bool:
        return bool(
            self.rows
            and self.regimes
            and self.primary_frame_count
            and not self.blocking_issues
            and all(regime.final_status is not None for regime in self.regimes)
        )


@dataclass(slots=True)
class MetricSet:
    record_count: int
    mae_fuel: float | None
    mape_fuel: float | None
    mape_fuel_count: int
    mae_steam: float | None
    mape_steam: float | None
    mape_steam_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "mae_fuel_g_h": self.mae_fuel,
            "mape_fuel_percent": self.mape_fuel,
            "mape_fuel_record_count": self.mape_fuel_count,
            "mae_steam_g_h": self.mae_steam,
            "mape_steam_percent": self.mape_steam,
            "mape_steam_record_count": self.mape_steam_count,
        }


@dataclass(slots=True)
class StageResult:
    scope: str
    scope_id: str
    stage: str
    metrics: MetricSet
    start_time: datetime | None
    end_time: datetime | None


@dataclass(slots=True)
class SelectedFrame:
    row: PredictionRow
    label: str
    rank: int
    score: float
    score_method: str
    abs_error_fuel: float
    ape_fuel: float | None
    abs_error_steam: float
    ape_steam: float | None


@dataclass(slots=True)
class AnalysisResult:
    experiment: ExperimentData
    overall_metrics: MetricSet
    trained_untrained_metrics: dict[bool, MetricSet]
    regime_metrics: dict[str, MetricSet]
    regime_stages: dict[str, list[StageResult]]
    overall_stages: list[StageResult]
    group_stages: dict[bool, list[StageResult]]
    row_metrics: dict[str, dict[str, float | None]]
    selected_frames: list[SelectedFrame]
    regime_conclusions: dict[str, str]
    overall_conclusion: str
    error_conclusions: dict[str, str]
    threshold_percent: float
    processing_notes: list[str] = field(default_factory=list)


def format_flow(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def status_label(value: bool) -> str:
    return "Обученный" if value else "Необученный"

