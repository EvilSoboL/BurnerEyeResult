from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image as PillowImage

from .models import ExperimentData, PredictionRow, Regime, ValidationIssue, VideoIndexEntry, VideoSource


REQUIRED_COLUMNS = {
    "timestamp",
    "frame_id",
    "unit",
    "expected_fuel_flow",
    "expected_diluent_flow",
    "trained_regime",
    "predicted_fuel_flow",
    "predicted_diluent_flow",
    "regime",
    "regime_confidence",
}

UNIT_FACTORS = {
    "kg/h": 1000.0,
    "kg/ч": 1000.0,
    "кг/h": 1000.0,
    "кг/ч": 1000.0,
    "g/h": 1.0,
    "g/ч": 1.0,
    "г/h": 1.0,
    "г/ч": 1.0,
    "g/s": 3600.0,
    "g/с": 3600.0,
    "г/s": 3600.0,
    "г/с": 3600.0,
}

CANONICAL_UNITS = {
    1000.0: "kg/h",
    1.0: "g/h",
    3600.0: "g/s",
}


def load_experiment(root: str | Path) -> ExperimentData:
    experiment = ExperimentData(root=Path(root).expanduser().resolve())
    results_path = experiment.root / "results.csv"
    frames_path = experiment.root / "frames"

    if not experiment.root.is_dir():
        _issue(experiment, "error", "experiment_missing", "Папка эксперимента не найдена.", blocking=True)
        return experiment
    if not results_path.is_file():
        _issue(experiment, "error", "results_missing", "Файл results.csv не найден.", blocking=True)
        return experiment

    try:
        text_sample = results_path.read_text(encoding="utf-8-sig")[:8192]
        delimiter = _detect_delimiter(text_sample)
        stream = results_path.open("r", encoding="utf-8-sig", newline="")
    except (OSError, UnicodeError) as exc:
        _issue(
            experiment,
            "error",
            "results_unreadable",
            f"Не удалось прочитать results.csv: {exc}",
            blocking=True,
        )
        return experiment

    with stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        headers = {header.strip() for header in (reader.fieldnames or []) if header}
        missing = sorted(REQUIRED_COLUMNS - headers)
        if missing:
            _issue(
                experiment,
                "error",
                "missing_columns",
                "Отсутствуют обязательные столбцы: " + ", ".join(missing),
                blocking=True,
            )
            return experiment
        raw_rows = list(reader)

    experiment.total_csv_rows = len(raw_rows)
    if not raw_rows:
        _issue(experiment, "error", "empty_csv", "results.csv не содержит строк.", blocking=True)
        return experiment

    frame_map: dict[str, Path] = {}
    if frames_path.is_dir():
        for candidate in frames_path.iterdir():
            if candidate.is_file() and candidate.suffix.casefold() in {".jpg", ".jpeg"}:
                frame_map[candidate.stem.casefold()] = candidate
    else:
        _issue(
            experiment,
            "error",
            "frames_missing",
            "Папка frames не найдена. Для раздела с кадрами нужен хотя бы один JPEG.",
            blocking=True,
        )

    seen_frame_ids: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        try:
            parsed = _parse_row(raw, row_number)
        except ValueError as exc:
            experiment.excluded_rows += 1
            _issue(
                experiment,
                "warning",
                "row_excluded",
                f"Строка исключена: {exc}",
                row_number=row_number,
            )
            continue

        if any(
            value is not None and value < 0
            for value in (
                parsed.expected_fuel,
                parsed.expected_steam,
                parsed.predicted_fuel,
                parsed.predicted_steam,
            )
        ):
            _issue(
                experiment,
                "warning",
                "negative_flow",
                "Строка содержит отрицательный расход. Значение сохранено для расчётов.",
                row_number=row_number,
            )
        if (
            parsed.regime_confidence is not None
            and not 0 <= parsed.regime_confidence <= 1
        ):
            _issue(
                experiment,
                "warning",
                "confidence_range",
                "regime_confidence находится вне ожидаемого диапазона от 0 до 1.",
                row_number=row_number,
            )

        normalized_id = parsed.frame_id.casefold()
        if normalized_id in seen_frame_ids:
            experiment.excluded_rows += 1
            _issue(
                experiment,
                "warning",
                "duplicate_frame_id",
                f"Строка исключена: frame_id «{parsed.frame_id}» встречается повторно.",
                row_number=row_number,
            )
            continue
        seen_frame_ids.add(normalized_id)

        image_path = frame_map.get(normalized_id)
        if image_path is not None and not _image_is_readable(image_path):
            _issue(
                experiment,
                "warning",
                "corrupt_frame",
                f"Кадр {image_path.name} повреждён или имеет неподдерживаемый формат.",
                row_number=row_number,
            )
            image_path = None
        parsed.image_path = image_path
        parsed.temporal_frame_count = _inspect_temporal_window(
            frames_path / parsed.frame_id, experiment, row_number, parsed.frame_id
        )
        if parsed.temporal_frame_count:
            experiment.temporal_window_count += 1
        experiment.rows.append(parsed)

    if not experiment.rows:
        _issue(
            experiment,
            "error",
            "no_valid_rows",
            "После проверки не осталось корректных строк с результатами.",
            blocking=True,
        )
        return experiment

    if experiment.predicted_fuel_count == 0:
        _issue(experiment, "info", "channel_unavailable", "В эксперименте нет прогнозов топлива.")
    if experiment.predicted_steam_count == 0:
        _issue(experiment, "info", "channel_unavailable", "В эксперименте нет прогнозов пара.")

    _load_video_sources(experiment)
    _load_video_index(experiment)
    _validate_video_provenance(experiment)

    experiment.rows.sort(key=lambda row: row.timestamp)
    _build_regimes(experiment)

    used_ids = {row.frame_id.casefold() for row in experiment.rows}
    experiment.orphan_frames = sorted(
        (path for key, path in frame_map.items() if key not in used_ids),
        key=lambda path: path.name.casefold(),
    )
    if experiment.orphan_frames:
        names = ", ".join(path.name for path in experiment.orphan_frames[:10])
        suffix = "…" if len(experiment.orphan_frames) > 10 else ""
        _issue(
            experiment,
            "warning",
            "orphan_frames",
            f"Найдено несвязанных основных кадров: {len(experiment.orphan_frames)} ({names}{suffix}).",
        )

    missing_frames = [row.frame_id for row in experiment.rows if row.image_path is None]
    if missing_frames:
        sample = ", ".join(missing_frames[:10])
        suffix = "…" if len(missing_frames) > 10 else ""
        _issue(
            experiment,
            "warning",
            "missing_primary_frames",
            f"Нет основного JPEG для {len(missing_frames)} строк ({sample}{suffix}). "
            "Эти строки участвуют в метриках, но не в выборе кадров.",
        )
    if experiment.primary_frame_count == 0:
        _issue(
            experiment,
            "error",
            "no_usable_frames",
            "Не найдено ни одного читаемого основного кадра; раздел лучших и худших кадров сформировать нельзя.",
            blocking=True,
        )
    elif experiment.primary_frame_count < 4:
        _issue(
            experiment,
            "warning",
            "few_frames",
            f"Доступно только {experiment.primary_frame_count} кадр(а/ов); в отчёт войдут все доступные.",
        )

    for regime in experiment.regimes:
        if regime.is_conflict:
            ones = sum(row.trained_regime for row in regime.rows)
            zeros = len(regime.rows) - ones
            _issue(
                experiment,
                "warning",
                "trained_status_conflict",
                f"{regime.display_name}: режим одновременно указан как обученный и необученный "
                f"(1 — {ones}, 0 — {zeros}). Выберите итоговый статус.",
            )

    zero_fuel = sum(row.expected_fuel_g_h == 0 for row in experiment.rows)
    zero_steam = sum(row.expected_steam_g_h == 0 for row in experiment.rows)
    if zero_fuel or zero_steam:
        _issue(
            experiment,
            "warning",
            "zero_expected",
            f"Строки с нулевым фактическим расходом: топливо — {zero_fuel}, пар — {zero_steam}. "
            "Они войдут в MAE, но будут исключены из соответствующего MAPE.",
        )
    return experiment


def _load_video_sources(experiment: ExperimentData) -> None:
    path = experiment.root / "source.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(experiment, "warning", "source_json_unreadable", f"Не удалось прочитать source.json: {exc}")
        return
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        _issue(experiment, "warning", "source_json_schema", "В source.json ожидается массив videos.")
        return

    for number, item in enumerate(videos, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("source_video"), str) or not item["source_video"].strip():
            _issue(experiment, "warning", "source_video_invalid", f"Запись videos[{number - 1}] не содержит имени source_video.")
            continue
        expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
        window = item.get("window") if isinstance(item.get("window"), dict) else {}
        completed = _optional_nonnegative_int(item.get("completed_records"))
        if item.get("completed_records") is not None and completed is None:
            _issue(experiment, "warning", "source_completed_records_invalid", f"Некорректный completed_records у видео {item['source_video']}.")
        experiment.video_sources.append(VideoSource(
            source_video=item["source_video"],
            status=str(item.get("status", "")),
            stop_reason=str(item["stop_reason"]) if item.get("stop_reason") is not None else None,
            completed_records=completed,
            fps=_optional_finite(item.get("fps")),
            width=_optional_nonnegative_int(item.get("width")),
            height=_optional_nonnegative_int(item.get("height")),
            frame_count=_optional_nonnegative_int(item.get("frame_count")),
            model_path=str(item["model_path"]) if item.get("model_path") is not None else None,
            window_num_frames=_optional_nonnegative_int(window.get("num_frames")),
            clip_duration_s=_optional_finite(window.get("clip_duration_s")),
            expected_fuel=_optional_finite(expected.get("fuel_flow")),
            expected_steam=_optional_finite(expected.get("diluent_flow")),
            expected_unit=str(expected["unit"]) if expected.get("unit") is not None else None,
            fuel_type=str(expected["fuel_type"]) if expected.get("fuel_type") is not None else None,
            diluent_type=str(expected["diluent_type"]) if expected.get("diluent_type") is not None else None,
            trained_regime=expected.get("trained_regime") if isinstance(expected.get("trained_regime"), bool) else None,
        ))
        if str(item.get("status", "")).casefold() == "error":
            reason = str(item.get("stop_reason") or "Причина не указана")
            _issue(experiment, "warning", "source_video_partial", f"Видео {item['source_video']} завершилось с ошибкой: {reason} Сохранено записей: {completed if completed is not None else 'не указано'}.")

    declared = sum(video.completed_records or 0 for video in experiment.video_sources)
    if any(video.completed_records is not None for video in experiment.video_sources) and declared != experiment.total_csv_rows:
        _issue(experiment, "warning", "source_completed_records_mismatch", f"Сумма completed_records ({declared}) не совпадает с числом строк CSV ({experiment.total_csv_rows}).")


def _load_video_index(experiment: ExperimentData) -> None:
    path = experiment.root / "video_index.csv"
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"frame_id", "source_video", "start_s", "end_s", "source_frame_indices"}
            headers = {str(header).strip() for header in (reader.fieldnames or []) if header}
            missing = required - headers
            if missing:
                _issue(experiment, "warning", "video_index_schema", "В video_index.csv отсутствуют столбцы: " + ", ".join(sorted(missing)))
                return
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        _issue(experiment, "warning", "video_index_unreadable", f"Не удалось прочитать video_index.csv: {exc}")
        return

    counts: dict[str, int] = {}
    for raw in raw_rows:
        key = str(raw.get("frame_id") or "").strip().casefold()
        if key:
            counts[key] = counts.get(key, 0) + 1
    duplicates = {key for key, count in counts.items() if count > 1}
    if duplicates:
        _issue(experiment, "warning", "video_index_duplicate_id", f"В video_index.csv дублированы frame_id: {len(duplicates)}; эти связи пропущены.")

    row_map = {row.frame_id.casefold(): row for row in experiment.rows}
    linked: set[str] = set()
    for line_number, raw in enumerate(raw_rows, start=2):
        values = {str(key).strip(): (value or "").strip() for key, value in raw.items() if key}
        key = values.get("frame_id", "").casefold()
        if not key or key in duplicates:
            continue
        if key not in row_map:
            continue
        try:
            source_video = values["source_video"]
            if not source_video:
                raise ValueError("пустой source_video")
            start_s = _parse_finite(values["start_s"], "start_s")
            end_s = _parse_finite(values["end_s"], "end_s")
            if start_s < 0 or end_s < 0 or end_s <= start_s:
                raise ValueError("интервал должен быть неотрицательным и end_s должен быть больше start_s")
            text_indices = values["source_frame_indices"]
            parts = text_indices.split(";") if text_indices else []
            if not parts or any(not re.fullmatch(r"\d+", part) for part in parts):
                raise ValueError("source_frame_indices должен содержать целые числа через ;")
            indices = [int(part) for part in parts]
            if any(right <= left for left, right in zip(indices, indices[1:])):
                raise ValueError("source_frame_indices должны строго возрастать")
            row_map[key].video_metadata = VideoIndexEntry(source_video, start_s, end_s, indices)
            linked.add(key)
        except (ValueError, KeyError) as exc:
            _issue(experiment, "warning", "video_index_record_invalid", f"Запись video_index.csv, строка {line_number}, пропущена: {exc}")

    missing = set(row_map) - linked
    if missing:
        _issue(experiment, "warning", "video_index_missing_ids", f"Для {len(missing)} строк results.csv нет корректной уникальной связи в video_index.csv.")
    extra = set(counts) - set(row_map)
    if extra:
        _issue(experiment, "warning", "video_index_extra_ids", f"В video_index.csv найдено frame_id без строки results.csv: {len(extra)}.")


def _validate_video_provenance(experiment: ExperimentData) -> None:
    sources_by_name: dict[str, list[VideoSource]] = {}
    for source in experiment.video_sources:
        sources_by_name.setdefault(source.source_video.casefold(), []).append(source)
    for source_video, sources in sources_by_name.items():
        if len(sources) > 1:
            _issue(experiment, "warning", "source_video_duplicate", f"source.json содержит повторные записи для {sources[0].source_video}; связь метаданных неоднозначна.")
    for row in experiment.rows:
        entry = row.video_metadata
        if entry is None:
            continue
        candidates = sources_by_name.get(entry.source_video.casefold(), [])
        if len(candidates) != 1:
            continue
        source = candidates[0]
        if source.window_num_frames is not None and row.temporal_frame_count and source.window_num_frames != row.temporal_frame_count:
            _issue(experiment, "warning", "source_window_mismatch", f"Число кадров окна для {entry.source_video} не совпадает с frames/{row.frame_id}.")
        if source.clip_duration_s is not None and abs((entry.end_s - entry.start_s) - source.clip_duration_s) > 1e-6:
            _issue(experiment, "warning", "source_window_duration_mismatch", f"Длительность окна {row.frame_id} не совпадает с source.json для {entry.source_video}.")
        try:
            factor = _parse_unit(source.expected_unit or "")[1]
        except ValueError:
            continue
        if source.expected_fuel is not None and abs(source.expected_fuel * factor - row.expected_fuel_g_h) > 1e-6:
            _issue(experiment, "warning", "source_expected_mismatch", f"Фактическая цель топлива в results.csv расходится с source.json для {entry.source_video}.")
        if source.expected_steam is not None and abs(source.expected_steam * factor - row.expected_steam_g_h) > 1e-6:
            _issue(experiment, "warning", "source_expected_mismatch", f"Фактическая цель пара в results.csv расходится с source.json для {entry.source_video}.")
        if source.trained_regime is not None and source.trained_regime != row.trained_regime:
            _issue(experiment, "warning", "source_trained_mismatch", f"trained_regime в results.csv расходится с source.json для {entry.source_video}.")


def _optional_finite(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and str(parsed) == str(value) else None


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _parse_row(raw: dict[str, str], row_number: int) -> PredictionRow:
    values = {str(key).strip(): (value or "").strip() for key, value in raw.items() if key}
    timestamp = _parse_timestamp(values["timestamp"])
    frame_id = values["frame_id"]
    if not frame_id:
        raise ValueError("пустой frame_id")
    if frame_id in {".", ".."} or "/" in frame_id or "\\" in frame_id:
        raise ValueError("frame_id не должен содержать части пути")
    unit, factor = _parse_unit(values["unit"])

    expected_fuel = _parse_finite(values["expected_fuel_flow"], "expected_fuel_flow")
    expected_steam = _parse_finite(values["expected_diluent_flow"], "expected_diluent_flow")
    predicted_fuel = _parse_optional_finite(values["predicted_fuel_flow"], "predicted_fuel_flow")
    predicted_steam = _parse_optional_finite(values["predicted_diluent_flow"], "predicted_diluent_flow")
    if predicted_fuel is None and predicted_steam is None:
        raise ValueError("оба прогноза пусты")

    trained_text = values["trained_regime"]
    if trained_text not in {"0", "1"}:
        raise ValueError("trained_regime должен быть равен 0 или 1")
    confidence_text = values["regime_confidence"]
    confidence = (
        _parse_finite(confidence_text, "regime_confidence") if confidence_text else None
    )

    return PredictionRow(
        timestamp=timestamp,
        frame_id=frame_id,
        source_unit=unit,
        expected_fuel=expected_fuel,
        expected_steam=expected_steam,
        predicted_fuel=predicted_fuel,
        predicted_steam=predicted_steam,
        expected_fuel_g_h=expected_fuel * factor,
        expected_steam_g_h=expected_steam * factor,
        predicted_fuel_g_h=predicted_fuel * factor if predicted_fuel is not None else None,
        predicted_steam_g_h=predicted_steam * factor if predicted_steam is not None else None,
        trained_regime=trained_text == "1",
        predicted_regime=values["regime"],
        regime_confidence=confidence,
        source_row_number=row_number,
    )


def _parse_timestamp(value: str) -> datetime:
    if not value:
        raise ValueError("пустой timestamp")
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"):
            try:
                parsed = datetime.strptime(normalized, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ValueError(f"некорректный timestamp «{value}»")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_unit(value: str) -> tuple[str, float]:
    normalized = value.strip().casefold().replace(" ", "")
    factor = UNIT_FACTORS.get(normalized)
    if factor is None:
        raise ValueError(f"неподдерживаемая единица измерения «{value}»")
    return CANONICAL_UNITS[factor], factor


def _parse_finite(value: str, field_name: str) -> float:
    if not value:
        raise ValueError(f"пустое значение {field_name}")
    try:
        parsed = float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"{field_name} не является числом") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} должно быть конечным числом")
    return parsed


def _parse_optional_finite(value: str, field_name: str) -> float | None:
    if not value:
        return None
    return _parse_finite(value, field_name)


def _image_is_readable(path: Path) -> bool:
    try:
        with PillowImage.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _inspect_temporal_window(
    window_path: Path,
    experiment: ExperimentData,
    row_number: int,
    frame_id: str,
) -> int:
    if not window_path.is_dir():
        return 0
    frames = sorted(
        path
        for path in window_path.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".jpg", ".jpeg"}
        and path.stem.casefold().startswith("frame_")
    )
    if not frames:
        _issue(
            experiment,
            "warning",
            "empty_temporal_window",
            f"Папка временного окна frames/{frame_id} не содержит кадров frame_*.jpg.",
            row_number=row_number,
        )
        return 0
    unreadable = sum(not _image_is_readable(frame) for frame in frames)
    if unreadable:
        _issue(
            experiment,
            "warning",
            "corrupt_temporal_frames",
            f"Во временном окне frames/{frame_id} нечитаемых кадров: {unreadable}.",
            row_number=row_number,
        )
    numbers: list[int] = []
    for frame in frames:
        match = re.fullmatch(r"frame_(\d+)", frame.stem, flags=re.IGNORECASE)
        if match:
            numbers.append(int(match.group(1)))
    if numbers and numbers != list(range(min(numbers), max(numbers) + 1)):
        _issue(
            experiment,
            "warning",
            "temporal_window_gap",
            f"Во временном окне frames/{frame_id} нарушена последовательность номеров кадров.",
            row_number=row_number,
        )
    return len(frames)


def _build_regimes(experiment: ExperimentData) -> None:
    grouped: dict[tuple[float, float], list[PredictionRow]] = {}
    for row in experiment.rows:
        key = (round(row.expected_fuel_g_h, 6), round(row.expected_steam_g_h, 6))
        grouped.setdefault(key, []).append(row)
    for index, key in enumerate(sorted(grouped, key=lambda item: (item[1], item[0])), start=1):
        rows = sorted(grouped[key], key=lambda row: row.timestamp)
        regime = Regime(
            regime_id=f"R{index:03d}",
            fuel_g_h=key[0],
            steam_g_h=key[1],
            rows=rows,
        )
        if not regime.is_conflict:
            regime.final_status = regime.source_status
        for row in rows:
            row.regime_id = regime.regime_id
        experiment.regimes.append(regime)


def _issue(
    experiment: ExperimentData,
    severity: str,
    code: str,
    message: str,
    *,
    row_number: int | None = None,
    blocking: bool = False,
) -> None:
    experiment.issues.append(
        ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            row_number=row_number,
            blocking=blocking,
        )
    )
