from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .comparison import ComparisonAnalysis, analyze_comparison, metric_winner
from .comparison_reporting import generate_comparison_report
from .loader import load_experiment
from .models import ExperimentData, Regime
from .reporting import generate_report


TRAINED_LABEL = "Был в обучающей выборке"
UNTRAINED_LABEL = "Не был в обучающей выборке"


def resolve_experiment_directory(path: str | Path) -> Path:
    """Resolve an explicit experiment or its unique experiment child (at most two levels)."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or (root / "results.csv").is_file():
        return root

    def is_report_directory(candidate: Path) -> bool:
        name = candidate.name.casefold()
        return name.startswith(("prediction_report_", "model_comparison_"))

    candidates: set[Path] = set()
    try:
        children = [entry for entry in root.iterdir() if entry.is_dir() and not is_report_directory(entry)]
        for child in children:
            if (child / "results.csv").is_file():
                candidates.add(child.resolve())
            try:
                grandchildren = [entry for entry in child.iterdir() if entry.is_dir() and not is_report_directory(entry)]
            except OSError:
                continue
            for candidate in grandchildren:
                if (candidate / "results.csv").is_file():
                    candidates.add(candidate.resolve())
    except OSError:
        return root

    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        listing = "\n".join(f"• {candidate}" for candidate in sorted(candidates, key=lambda item: str(item).casefold()))
        raise ValueError(
            "В выбранной папке найдено несколько экспериментов. Выберите конкретную папку с results.csv:\n"
            + listing
        )
    return root


def load_selected_experiment(path: str | Path) -> ExperimentData:
    return load_experiment(resolve_experiment_directory(path))


class BurnerEyeReportApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BurnerEye - формирование PDF-отчёта")
        self.geometry("1360x860")
        self.minsize(1040, 680)
        self.configure(background="#F3F6FA")

        self.experiment: ExperimentData | None = None
        self.report_path: Path | None = None
        self.comparison: ComparisonAnalysis | None = None
        self.comparison_report_path: Path | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False

        self.path_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="5")
        self.status_var = tk.StringVar(value="Выберите папку эксперимента")
        self.result_var = tk.StringVar(value="")
        self.summary_vars = {
            "rows": tk.StringVar(value="0"),
            "regimes": tk.StringVar(value="0"),
            "trained": tk.StringVar(value="0"),
            "untrained": tk.StringVar(value="0"),
            "conflicts": tk.StringVar(value="0"),
            "frames": tk.StringVar(value="0 / 0"),
            "fuel_predictions": tk.StringVar(value="0"),
            "steam_predictions": tk.StringVar(value="0"),
        }
        self.comparison_first_path_var = tk.StringVar()
        self.comparison_second_path_var = tk.StringVar()
        self.comparison_status_var = tk.StringVar(
            value="Выберите две папки с результатами предсказаний"
        )
        self.comparison_result_var = tk.StringVar(value="")
        self.comparison_summary_vars = {
            "common": tk.StringVar(value="0"),
            "first_rows": tk.StringVar(value="0"),
            "second_rows": tk.StringVar(value="0"),
            "excluded": tk.StringVar(value="0 / 0"),
            "targets": tk.StringVar(value="—"),
        }

        self._configure_styles()
        self._build_ui()
        self.after(100, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#F3F6FA")
        style.configure("Card.TFrame", background="#FFFFFF")
        style.configure("Title.TLabel", background="#F3F6FA", foreground="#172033", font=("Segoe UI Semibold", 20))
        style.configure("Subtitle.TLabel", background="#F3F6FA", foreground="#64748B")
        style.configure("CardTitle.TLabel", background="#FFFFFF", foreground="#172033", font=("Segoe UI Semibold", 11))
        style.configure("Metric.TLabel", background="#FFFFFF", foreground="#0F6CBD", font=("Segoe UI Semibold", 18))
        style.configure("MetricCaption.TLabel", background="#FFFFFF", foreground="#64748B", font=("Segoe UI", 8))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), foreground="#FFFFFF", background="#0F6CBD", padding=(14, 8))
        style.map("Accent.TButton", background=[("active", "#0B5A9E"), ("disabled", "#AEBECD")])
        style.configure("Secondary.TButton", padding=(10, 7))
        style.configure("Treeview", rowheight=28, background="#FFFFFF", fieldbackground="#FFFFFF", borderwidth=0)
        style.configure("Treeview.Heading", background="#E9EFF6", foreground="#26364A", font=("Segoe UI Semibold", 9), padding=(5, 7))
        style.map("Treeview", background=[("selected", "#DCECFB")], foreground=[("selected", "#172033")])
        style.configure("Horizontal.TProgressbar", background="#0F6CBD", troughcolor="#DDE6EF")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(22, 18, 22, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Отчёт по предсказаниям BurnerEye", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Проверка эксперимента, корректировка режимов, MAE/MAPE, графики и лучшие/худшие кадры",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        workspace_notebook = ttk.Notebook(self)
        workspace_notebook.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=22,
            pady=(0, 18),
        )
        report_workspace = ttk.Frame(workspace_notebook)
        comparison_workspace = ttk.Frame(workspace_notebook)
        workspace_notebook.add(report_workspace, text="Отчёт по одной модели")
        workspace_notebook.add(comparison_workspace, text="Сравнение моделей")

        report_workspace.columnconfigure(0, weight=1)
        report_workspace.rowconfigure(2, weight=1)

        chooser = ttk.Frame(report_workspace, style="Card.TFrame", padding=14)
        chooser.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        chooser.columnconfigure(0, weight=1)
        ttk.Entry(chooser, textvariable=self.path_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.check_button = ttk.Button(
            chooser,
            text="Проверить путь",
            style="Secondary.TButton",
            command=self._start_loading,
        )
        self.check_button.grid(row=0, column=1, padx=(0, 8))
        self.choose_button = ttk.Button(
            chooser,
            text="Выбрать папку эксперимента",
            style="Accent.TButton",
            command=self._choose_folder,
        )
        self.choose_button.grid(row=0, column=2)

        summary = ttk.Frame(report_workspace, padding=(10, 0, 10, 10))
        summary.grid(row=1, column=0, sticky="ew")
        for index in range(len(self.summary_vars)):
            summary.columnconfigure(index, weight=1, uniform="summary")
        cards = [
            ("rows", "корректных записей"),
            ("regimes", "режимов"),
            ("trained", "обученных"),
            ("untrained", "необученных"),
            ("conflicts", "неразрешённых конфликтов"),
            ("frames", "кадров / строк"),
            ("fuel_predictions", "прогнозов топлива"),
            ("steam_predictions", "прогнозов пара"),
        ]
        for index, (key, caption) in enumerate(cards):
            card = ttk.Frame(summary, style="Card.TFrame", padding=(12, 9))
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 4, 0 if index == len(cards) - 1 else 4))
            ttk.Label(card, textvariable=self.summary_vars[key], style="Metric.TLabel").pack(anchor="w")
            ttk.Label(card, text=caption, style="MetricCaption.TLabel").pack(anchor="w")

        notebook = ttk.Notebook(report_workspace)
        notebook.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        regimes_tab = ttk.Frame(notebook, padding=10)
        issues_tab = ttk.Frame(notebook, padding=10)
        notebook.add(regimes_tab, text="Режимы")
        notebook.add(issues_tab, text="Ошибки и предупреждения")

        regimes_tab.columnconfigure(0, weight=1)
        regimes_tab.rowconfigure(1, weight=1)
        regime_toolbar = ttk.Frame(regimes_tab)
        regime_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            regime_toolbar,
            text="Выберите режимы и назначьте итоговый статус. Для конфликтов выбор обязателен.",
            style="Subtitle.TLabel",
        ).pack(side="left")
        self.set_untrained_button = ttk.Button(
            regime_toolbar,
            text=UNTRAINED_LABEL,
            style="Secondary.TButton",
            command=lambda: self._set_selected_status(False),
        )
        self.set_untrained_button.pack(side="right")
        self.set_trained_button = ttk.Button(
            regime_toolbar,
            text=TRAINED_LABEL,
            style="Secondary.TButton",
            command=lambda: self._set_selected_status(True),
        )
        self.set_trained_button.pack(side="right", padx=(0, 8))

        table_frame = ttk.Frame(regimes_tab, style="Card.TFrame")
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("regime", "count", "first", "last", "source", "final", "check")
        self.regime_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
        )
        headings = {
            "regime": "Режим",
            "count": "Записей",
            "first": "Первый кадр",
            "last": "Последний кадр",
            "source": "Исходный статус",
            "final": "Статус для отчёта",
            "check": "Проверка",
        }
        widths = {
            "regime": 280,
            "count": 70,
            "first": 130,
            "last": 130,
            "source": 170,
            "final": 170,
            "check": 110,
        }
        for column in columns:
            self.regime_tree.heading(column, text=headings[column])
            self.regime_tree.column(
                column,
                width=widths[column],
                minwidth=60,
                stretch=column == "regime",
                anchor="w" if column == "regime" else "center",
            )
        self.regime_tree.tag_configure("conflict", background="#FFF0E2")
        self.regime_tree.tag_configure("changed", background="#E9F5FF")
        self.regime_tree.grid(row=0, column=0, sticky="nsew")
        regime_scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.regime_tree.yview)
        regime_scroll_y.grid(row=0, column=1, sticky="ns")
        regime_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.regime_tree.xview)
        regime_scroll_x.grid(row=1, column=0, sticky="ew")
        self.regime_tree.configure(yscrollcommand=regime_scroll_y.set, xscrollcommand=regime_scroll_x.set)
        self.regime_tree.bind("<Double-1>", self._toggle_double_clicked_regime)

        issues_tab.columnconfigure(0, weight=1)
        issues_tab.rowconfigure(0, weight=1)
        issue_columns = ("severity", "row", "message")
        self.issue_tree = ttk.Treeview(issues_tab, columns=issue_columns, show="headings")
        self.issue_tree.heading("severity", text="Уровень")
        self.issue_tree.heading("row", text="Строка CSV")
        self.issue_tree.heading("message", text="Сообщение")
        self.issue_tree.column("severity", width=130, stretch=False)
        self.issue_tree.column("row", width=100, stretch=False, anchor="center")
        self.issue_tree.column("message", width=800, stretch=True)
        self.issue_tree.tag_configure("error", background="#FFE9EC")
        self.issue_tree.tag_configure("warning", background="#FFF5E6")
        self.issue_tree.tag_configure("info", background="#EAF3FC")
        self.issue_tree.grid(row=0, column=0, sticky="nsew")
        issue_scroll = ttk.Scrollbar(issues_tab, orient="vertical", command=self.issue_tree.yview)
        issue_scroll.grid(row=0, column=1, sticky="ns")
        self.issue_tree.configure(yscrollcommand=issue_scroll.set)

        footer = ttk.Frame(report_workspace, style="Card.TFrame", padding=(16, 12))
        footer.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        footer.columnconfigure(1, weight=1)
        ttk.Label(footer, text="Порог существенного изменения, %:", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.threshold_spinbox = ttk.Spinbox(
            footer,
            from_=0,
            to=100,
            increment=0.5,
            textvariable=self.threshold_var,
            width=8,
        )
        self.threshold_spinbox.grid(row=0, column=1, sticky="w", padx=(8, 16))
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, value=0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=(0, 12))
        footer.columnconfigure(2, weight=2)
        self.report_button = ttk.Button(
            footer,
            text="Сформировать PDF-отчёт",
            style="Accent.TButton",
            command=self._start_report,
            state="disabled",
        )
        self.report_button.grid(row=0, column=3)

        ttk.Label(footer, textvariable=self.status_var, background="#FFFFFF", foreground="#425466").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        self.open_button = ttk.Button(
            footer,
            text="Открыть папку отчёта",
            style="Secondary.TButton",
            command=self._open_report_folder,
            state="disabled",
        )
        self.open_button.grid(row=1, column=3, sticky="e", pady=(8, 0))
        ttk.Entry(footer, textvariable=self.result_var, state="readonly").grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

        self._build_comparison_tab(comparison_workspace)
        self._set_edit_controls(False)

    def _build_comparison_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        chooser = ttk.Frame(tab, style="Card.TFrame", padding=14)
        chooser.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        chooser.columnconfigure(1, weight=1)
        ttk.Label(
            chooser,
            text="Модель 1:",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        ttk.Entry(
            chooser,
            textvariable=self.comparison_first_path_var,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.comparison_first_choose_button = ttk.Button(
            chooser,
            text="Выбрать папку",
            style="Secondary.TButton",
            command=lambda: self._choose_comparison_folder(1),
        )
        self.comparison_first_choose_button.grid(
            row=0,
            column=2,
            padx=(0, 12),
            pady=(0, 8),
        )

        ttk.Label(
            chooser,
            text="Модель 2:",
            style="CardTitle.TLabel",
        ).grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(
            chooser,
            textvariable=self.comparison_second_path_var,
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self.comparison_second_choose_button = ttk.Button(
            chooser,
            text="Выбрать папку",
            style="Secondary.TButton",
            command=lambda: self._choose_comparison_folder(2),
        )
        self.comparison_second_choose_button.grid(
            row=1,
            column=2,
            padx=(0, 12),
        )
        self.compare_button = ttk.Button(
            chooser,
            text="Проверить и сравнить",
            style="Accent.TButton",
            command=self._start_comparison,
        )
        self.compare_button.grid(
            row=0,
            column=3,
            rowspan=2,
            sticky="ns",
        )

        summary = ttk.Frame(tab, padding=(10, 0, 10, 10))
        summary.grid(row=1, column=0, sticky="ew")
        for index in range(5):
            summary.columnconfigure(index, weight=1, uniform="comparison_summary")
        cards = [
            ("common", "общих режимов"),
            ("first_rows", "строк модели 1 в общих режимах"),
            ("second_rows", "строк модели 2 в общих режимах"),
            ("excluded", "исключено режимов: модель 1 / 2"),
            ("targets", "общих прогнозируемых каналов"),
        ]
        for index, (key, caption) in enumerate(cards):
            card = ttk.Frame(summary, style="Card.TFrame", padding=(12, 9))
            card.grid(
                row=0,
                column=index,
                sticky="nsew",
                padx=(0 if index == 0 else 4, 0 if index == len(cards) - 1 else 4),
            )
            ttk.Label(
                card,
                textvariable=self.comparison_summary_vars[key],
                style="Metric.TLabel",
            ).pack(anchor="w")
            ttk.Label(
                card,
                text=caption,
                style="MetricCaption.TLabel",
            ).pack(anchor="w")

        table_container = ttk.Frame(tab, padding=(10, 0, 10, 10))
        table_container.grid(row=2, column=0, sticky="nsew")
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(1, weight=1)
        ttk.Label(
            table_container,
            text=(
                "В таблице, PDF и CSV учитываются только режимы, "
                "присутствующие у обеих моделей."
            ),
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        table_frame = ttk.Frame(table_container, style="Card.TFrame")
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = (
            "regime", "first_rows", "second_rows",
            "fuel_first", "fuel_second", "fuel_winner",
            "steam_first", "steam_second", "steam_winner",
        )
        self.comparison_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
        )
        headings = {
            "regime": "Общий режим",
            "first_rows": "Строк модели 1",
            "second_rows": "Строк модели 2",
            "fuel_first": "Топливо, модель 1",
            "fuel_second": "Топливо, модель 2",
            "fuel_winner": "Итог по топливу",
            "steam_first": "Пар, модель 1",
            "steam_second": "Пар, модель 2",
            "steam_winner": "Итог по пару",
        }
        widths = {
            "regime": 330,
            "first_rows": 110,
            "second_rows": 110,
            "fuel_first": 120,
            "fuel_second": 120,
            "fuel_winner": 170,
            "steam_first": 120,
            "steam_second": 120,
            "steam_winner": 170,
        }
        for column in columns:
            self.comparison_tree.heading(column, text=headings[column])
            self.comparison_tree.column(
                column,
                width=widths[column],
                minwidth=80,
                stretch=column in {"regime", "fuel_winner", "steam_winner"},
                anchor="w" if column in {"regime", "fuel_winner", "steam_winner"} else "center",
            )
        self.comparison_tree.grid(row=0, column=0, sticky="nsew")
        scroll_y = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.comparison_tree.yview,
        )
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.comparison_tree.xview,
        )
        scroll_x.grid(row=1, column=0, sticky="ew")
        self.comparison_tree.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
        )

        footer = ttk.Frame(tab, style="Card.TFrame", padding=(16, 12))
        footer.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        footer.columnconfigure(0, weight=1)
        self.comparison_progress = ttk.Progressbar(
            footer,
            mode="determinate",
            maximum=100,
            value=0,
        )
        self.comparison_progress.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 12),
        )
        self.comparison_report_button = ttk.Button(
            footer,
            text="Сформировать сравнительный PDF",
            style="Accent.TButton",
            command=self._start_comparison_report,
            state="disabled",
        )
        self.comparison_report_button.grid(row=0, column=1)
        ttk.Label(
            footer,
            textvariable=self.comparison_status_var,
            background="#FFFFFF",
            foreground="#425466",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.comparison_open_button = ttk.Button(
            footer,
            text="Открыть папку отчёта",
            style="Secondary.TButton",
            command=self._open_comparison_report_folder,
            state="disabled",
        )
        self.comparison_open_button.grid(row=1, column=1, sticky="e", pady=(8, 0))
        ttk.Entry(
            footer,
            textvariable=self.comparison_result_var,
            state="readonly",
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Выберите папку эксперимента BurnerEye",
            initialdir=self.path_var.get() or str(Path.cwd()),
        )
        if selected:
            self.path_var.set(selected)
            self._start_loading()

    def _choose_comparison_folder(self, position: int) -> None:
        target = (
            self.comparison_first_path_var
            if position == 1
            else self.comparison_second_path_var
        )
        selected = filedialog.askdirectory(
            title=f"Выберите папку результатов для модели {position}",
            initialdir=target.get() or str(Path.cwd()),
        )
        if selected:
            target.set(selected)

    def _start_loading(self) -> None:
        if self.busy:
            return
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Папка не выбрана", "Укажите папку эксперимента.")
            return
        self.busy = True
        self.experiment = None
        self.report_path = None
        self.result_var.set("")
        self.status_var.set("Проверка results.csv и кадров…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._set_busy_controls()

        def work() -> None:
            try:
                result = load_selected_experiment(path)
                self.events.put(("loaded", result))
            except Exception as exc:
                self.events.put(("load_error", exc))

        threading.Thread(target=work, daemon=True).start()

    def _start_comparison(self) -> None:
        if self.busy:
            return
        first_path = self.comparison_first_path_var.get().strip()
        second_path = self.comparison_second_path_var.get().strip()
        if not first_path or not second_path:
            messagebox.showwarning(
                "Папки не выбраны",
                "Укажите две папки с результатами предсказаний.",
            )
            return
        try:
            if Path(first_path).expanduser().resolve() == Path(
                second_path
            ).expanduser().resolve():
                raise ValueError(
                    "Для сравнения выберите две разные папки с результатами."
                )
        except OSError as exc:
            messagebox.showerror("Некорректный путь", str(exc))
            return

        self.busy = True
        self.comparison = None
        self.comparison_report_path = None
        self.comparison_result_var.set("")
        self.comparison_status_var.set(
            "Проверка results.csv и поиск общих режимов…"
        )
        self.comparison_progress.configure(mode="indeterminate")
        self.comparison_progress.start(12)
        self._clear_comparison()
        self._set_busy_controls()

        def work() -> None:
            try:
                resolved_first = resolve_experiment_directory(first_path)
                resolved_second = resolve_experiment_directory(second_path)
                if resolved_first == resolved_second:
                    raise ValueError(
                        "Обе папки разрешаются в один эксперимент. Выберите два разных эксперимента."
                    )
                first = load_experiment(resolved_first)
                second = load_experiment(resolved_second)
                result = analyze_comparison(first, second)
                self.events.put(("comparison_loaded", result))
            except Exception as exc:
                self.events.put(("comparison_load_error", exc))

        threading.Thread(target=work, daemon=True).start()

    def _start_report(self) -> None:
        if self.busy or self.experiment is None:
            return
        try:
            threshold = float(self.threshold_var.get().strip().replace(",", "."))
            if threshold < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Некорректный порог",
                "Введите неотрицательное число для порога существенного изменения.",
            )
            return
        if not self.experiment.can_generate:
            messagebox.showwarning(
                "Отчёт пока недоступен",
                "Устраните конфликты режимов и критические ошибки проверки.",
            )
            return
        self.busy = True
        self.report_path = None
        self.result_var.set("")
        self.status_var.set("Подготовка отчёта…")
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._set_busy_controls()

        def progress(value: int, text: str) -> None:
            self.events.put(("progress", (value, text)))

        def work() -> None:
            try:
                report_path = generate_report(
                    self.experiment,
                    threshold,
                    progress=progress,
                )
                self.events.put(("report_ready", report_path))
            except Exception as exc:
                self.events.put(("report_error", exc))

        threading.Thread(target=work, daemon=True).start()

    def _start_comparison_report(self) -> None:
        if self.busy or self.comparison is None:
            return
        self.busy = True
        self.comparison_report_path = None
        self.comparison_result_var.set("")
        self.comparison_status_var.set("Подготовка сравнительного отчёта…")
        self.comparison_progress.stop()
        self.comparison_progress.configure(mode="determinate", value=0)
        self._set_busy_controls()

        def progress(value: int, text: str) -> None:
            self.events.put(("comparison_progress", (value, text)))

        def work() -> None:
            try:
                report_path = generate_comparison_report(
                    self.comparison,
                    progress=progress,
                )
                self.events.put(("comparison_report_ready", report_path))
            except Exception as exc:
                self.events.put(("comparison_report_error", exc))

        threading.Thread(target=work, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "loaded":
                    self._finish_loading(payload)
                elif event == "load_error":
                    self._finish_error("Не удалось проверить эксперимент", payload)
                elif event == "progress":
                    value, text = payload
                    self.progress.configure(value=value)
                    self.status_var.set(str(text))
                elif event == "report_ready":
                    self._finish_report(payload)
                elif event == "report_error":
                    self._finish_error("Не удалось сформировать отчёт", payload)
                elif event == "comparison_loaded":
                    self._finish_comparison_loading(payload)
                elif event == "comparison_load_error":
                    self._finish_comparison_error(
                        "Не удалось сравнить модели",
                        payload,
                    )
                elif event == "comparison_progress":
                    value, text = payload
                    self.comparison_progress.configure(value=value)
                    self.comparison_status_var.set(str(text))
                elif event == "comparison_report_ready":
                    self._finish_comparison_report(payload)
                elif event == "comparison_report_error":
                    self._finish_comparison_error(
                        "Не удалось сформировать сравнительный отчёт",
                        payload,
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _finish_loading(self, experiment: object) -> None:
        assert isinstance(experiment, ExperimentData)
        try:
            selected_path = Path(self.path_var.get()).expanduser().resolve()
        except OSError:
            selected_path = experiment.root
        auto_resolved = experiment.root != selected_path
        self.path_var.set(str(experiment.root))
        self.busy = False
        self.experiment = experiment
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._populate_experiment()
        if experiment.blocking_issues:
            self.status_var.set(
                f"Проверка завершена: критических ошибок — {len(experiment.blocking_issues)}."
            )
        elif experiment.conflict_count:
            self.status_var.set(
                f"Проверка завершена. Прогнозы: топливо — {experiment.predicted_fuel_count}, "
                f"пар — {experiment.predicted_steam_count}. Разрешите конфликтов: {experiment.conflict_count}."
            )
        else:
            self.status_var.set(
                f"Проверка завершена. Прогнозы: топливо — {experiment.predicted_fuel_count}, "
                f"пар — {experiment.predicted_steam_count}. Эксперимент готов к отчёту."
            )
        if auto_resolved:
            self.status_var.set(self.status_var.get() + " Выбрана единственная вложенная папка с results.csv.")
        self._refresh_controls()

    def _finish_report(self, report_path: object) -> None:
        assert isinstance(report_path, Path)
        self.busy = False
        self.report_path = report_path
        self.progress.configure(value=100)
        self.status_var.set("PDF и расчётные CSV сформированы и проверены.")
        self.result_var.set(str(report_path))
        self._refresh_controls()
        messagebox.showinfo(
            "Отчёт готов",
            f"Отчёт сохранён:\n{report_path}",
        )

    def _finish_comparison_loading(self, comparison: object) -> None:
        assert isinstance(comparison, ComparisonAnalysis)
        self.busy = False
        self.comparison = comparison
        self.comparison_first_path_var.set(str(comparison.first.root))
        self.comparison_second_path_var.set(str(comparison.second.root))
        self.comparison_progress.stop()
        self.comparison_progress.configure(mode="determinate", value=0)
        self._populate_comparison()
        excluded_count = len(comparison.first_only_regimes) + len(
            comparison.second_only_regimes
        )
        self.comparison_status_var.set(
            f"Найдено общих режимов: {len(comparison.common_regimes)}. "
            f"Общие цели: {self._format_targets(comparison.comparable_targets)}. "
            f"Исключено режимов вне пересечения: {excluded_count}."
        )
        self._refresh_controls()

    def _finish_comparison_report(self, report_path: object) -> None:
        assert isinstance(report_path, Path)
        self.busy = False
        self.comparison_report_path = report_path
        self.comparison_progress.configure(value=100)
        self.comparison_status_var.set(
            "Сравнительный PDF и расчётные CSV сформированы и проверены."
        )
        self.comparison_result_var.set(str(report_path))
        self._refresh_controls()
        messagebox.showinfo(
            "Сравнительный отчёт готов",
            f"Отчёт сохранён:\n{report_path}",
        )

    def _finish_error(self, title: str, error: object) -> None:
        self.busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.status_var.set(str(error))
        self._refresh_controls()
        messagebox.showerror(title, str(error))

    def _finish_comparison_error(self, title: str, error: object) -> None:
        self.busy = False
        self.comparison_progress.stop()
        self.comparison_progress.configure(mode="determinate", value=0)
        self.comparison_status_var.set(str(error))
        self._refresh_controls()
        messagebox.showerror(title, str(error))

    def _populate_experiment(self) -> None:
        for item in self.regime_tree.get_children():
            self.regime_tree.delete(item)
        for item in self.issue_tree.get_children():
            self.issue_tree.delete(item)
        if self.experiment is None:
            return
        for regime in self.experiment.regimes:
            self._upsert_regime(regime)
        for index, issue in enumerate(self.experiment.issues):
            self.issue_tree.insert(
                "",
                "end",
                iid=f"issue_{index}",
                values=(
                    issue.severity_label,
                    issue.row_number or "",
                    issue.message,
                ),
                tags=(issue.severity,),
            )
        self._refresh_summary()

    def _clear_comparison(self) -> None:
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        for value in self.comparison_summary_vars.values():
            value.set("0")
        self.comparison_summary_vars["targets"].set("—")

    def _populate_comparison(self) -> None:
        self._clear_comparison()
        comparison = self.comparison
        if comparison is None:
            return
        for index, item in enumerate(comparison.common_regimes, start=1):
            fuel_first, fuel_second, fuel_result = self._format_comparison_channel(
                item.first_metrics, item.second_metrics, "fuel",
                comparison.first_name, comparison.second_name,
            )
            steam_first, steam_second, steam_result = self._format_comparison_channel(
                item.first_metrics, item.second_metrics, "steam",
                comparison.first_name, comparison.second_name,
            )
            self.comparison_tree.insert(
                "",
                "end",
                iid=f"common_{index}",
                values=(
                    item.display_name,
                    len(item.first_regime.rows),
                    len(item.second_regime.rows),
                    fuel_first,
                    fuel_second,
                    fuel_result,
                    steam_first,
                    steam_second,
                    steam_result,
                ),
            )
        self.comparison_summary_vars["common"].set(
            str(len(comparison.common_regimes))
        )
        self.comparison_summary_vars["first_rows"].set(
            str(comparison.first_overall_metrics.record_count)
        )
        self.comparison_summary_vars["second_rows"].set(
            str(comparison.second_overall_metrics.record_count)
        )
        self.comparison_summary_vars["excluded"].set(
            f"{len(comparison.first_only_regimes)} / "
            f"{len(comparison.second_only_regimes)}"
        )
        self.comparison_summary_vars["targets"].set(self._format_targets(comparison.comparable_targets))

    @staticmethod
    def _format_targets(targets: set[str]) -> str:
        labels = [label for key, label in (("fuel", "топливо"), ("steam", "пар")) if key in targets]
        return ", ".join(labels) if labels else "нет"

    @staticmethod
    def _format_comparison_channel(first, second, channel: str, first_name: str, second_name: str):
        first_mae = getattr(first, f"mae_{channel}")
        second_mae = getattr(second, f"mae_{channel}")
        first_count = getattr(first, f"mae_{channel}_count")
        second_count = getattr(second, f"mae_{channel}_count")
        if not first_count or not second_count:
            missing = "Нет сопоставимых данных"
            return missing, missing, missing
        first_mape = getattr(first, f"mape_{channel}")
        second_mape = getattr(second, f"mape_{channel}")
        if first_mape is not None and second_mape is not None:
            unit = "%"
            first_value, second_value = first_mape, second_mape
            winner = metric_winner(first_mape, second_mape, first_name, second_name)
        else:
            unit = "г/ч"
            first_value, second_value = first_mae, second_mae
            winner = metric_winner(first_mae, second_mae, first_name, second_name)
        return f"{first_value:.3f} {unit}", f"{second_value:.3f} {unit}", winner

    def _upsert_regime(self, regime: Regime) -> None:
        check_label = (
            "Исправлено"
            if regime.is_conflict and regime.final_status is not None
            else "Конфликт"
            if regime.is_conflict
            else "Корректно"
        )
        tags = (
            ("changed",)
            if regime.manually_changed
            else ("conflict",)
            if regime.is_conflict
            else ()
        )
        values = (
            regime.display_name,
            len(regime.rows),
            regime.first_timestamp.strftime("%d.%m.%Y %H:%M:%S"),
            regime.last_timestamp.strftime("%d.%m.%Y %H:%M:%S"),
            regime.source_status_label,
            regime.final_status_label,
            check_label,
        )
        if self.regime_tree.exists(regime.regime_id):
            self.regime_tree.item(regime.regime_id, values=values, tags=tags)
        else:
            self.regime_tree.insert("", "end", iid=regime.regime_id, values=values, tags=tags)

    def _set_selected_status(self, status: bool) -> None:
        if self.experiment is None or self.busy:
            return
        selected = self.regime_tree.selection()
        if not selected:
            messagebox.showinfo("Режим не выбран", "Выберите один или несколько режимов в таблице.")
            return
        lookup = {regime.regime_id: regime for regime in self.experiment.regimes}
        for regime_id in selected:
            regime = lookup.get(regime_id)
            if regime is not None:
                regime.final_status = status
                self._upsert_regime(regime)
        self._refresh_summary()
        self._refresh_controls()
        if self.experiment.conflict_count:
            self.status_var.set(
                f"Осталось разрешить конфликтов: {self.experiment.conflict_count}."
            )
        elif not self.experiment.blocking_issues:
            self.status_var.set("Все режимы классифицированы. Эксперимент готов к отчёту.")

    def _toggle_double_clicked_regime(self, event: tk.Event) -> None:
        if self.experiment is None or self.busy:
            return
        item = self.regime_tree.identify_row(event.y)
        if not item:
            return
        self.regime_tree.selection_set(item)
        lookup = {regime.regime_id: regime for regime in self.experiment.regimes}
        regime = lookup[item]
        self._set_selected_status(not bool(regime.final_status))

    def _refresh_summary(self) -> None:
        experiment = self.experiment
        if experiment is None:
            for value in self.summary_vars.values():
                value.set("0")
            return
        self.summary_vars["rows"].set(str(len(experiment.rows)))
        self.summary_vars["regimes"].set(str(len(experiment.regimes)))
        self.summary_vars["trained"].set(str(experiment.trained_regime_count))
        self.summary_vars["untrained"].set(str(experiment.untrained_regime_count))
        self.summary_vars["conflicts"].set(str(experiment.conflict_count))
        self.summary_vars["frames"].set(
            f"{experiment.primary_frame_count} / {len(experiment.rows)}"
        )
        self.summary_vars["fuel_predictions"].set(str(experiment.predicted_fuel_count))
        self.summary_vars["steam_predictions"].set(str(experiment.predicted_steam_count))

    def _set_busy_controls(self) -> None:
        self.choose_button.configure(state="disabled")
        self.check_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.comparison_first_choose_button.configure(state="disabled")
        self.comparison_second_choose_button.configure(state="disabled")
        self.compare_button.configure(state="disabled")
        self.comparison_report_button.configure(state="disabled")
        self.comparison_open_button.configure(state="disabled")
        self._set_edit_controls(False)

    def _refresh_controls(self) -> None:
        state = "disabled" if self.busy else "normal"
        self.choose_button.configure(state=state)
        self.check_button.configure(state=state)
        self.comparison_first_choose_button.configure(state=state)
        self.comparison_second_choose_button.configure(state=state)
        self.compare_button.configure(state=state)
        self._set_edit_controls(not self.busy and self.experiment is not None)
        self.report_button.configure(
            state=(
                "normal"
                if not self.busy
                and self.experiment is not None
                and self.experiment.can_generate
                else "disabled"
            )
        )
        self.comparison_report_button.configure(
            state=(
                "normal"
                if not self.busy and self.comparison is not None
                else "disabled"
            )
        )
        self.comparison_open_button.configure(
            state=(
                "normal"
                if not self.busy
                and self.comparison_report_path is not None
                and self.comparison_report_path.exists()
                else "disabled"
            )
        )
        self.open_button.configure(
            state=(
                "normal"
                if not self.busy
                and self.report_path is not None
                and self.report_path.exists()
                else "disabled"
            )
        )

    def _set_edit_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.set_trained_button.configure(state=state)
        self.set_untrained_button.configure(state=state)
        self.threshold_spinbox.configure(state=state)

    def _open_report_folder(self) -> None:
        if self.report_path is None:
            return
        try:
            os.startfile(str(self.report_path.parent))
        except OSError as exc:
            messagebox.showerror("Не удалось открыть папку", str(exc))

    def _open_comparison_report_folder(self) -> None:
        if self.comparison_report_path is None:
            return
        try:
            os.startfile(str(self.comparison_report_path.parent))
        except OSError as exc:
            messagebox.showerror("Не удалось открыть папку", str(exc))


def run() -> None:
    app = BurnerEyeReportApp()
    app.mainloop()
