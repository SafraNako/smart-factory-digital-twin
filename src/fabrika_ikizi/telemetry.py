"""Reading and writing the sensor stream as CSV.

Columns: ``zaman_saat`` (hours), ``yuk`` (load fraction 0-1, the PLC command), ``sicaklik_C``,
``titresim_mm_s``. Empty cells or ``nan`` are lost samples. A semicolon-delimited file with decimal commas
(what Turkish spreadsheet software writes) is read as well.
"""
from __future__ import annotations

import csv
import math

import numpy as np

COLUMNS = ("zaman_saat", "yuk", "sicaklik_C", "titresim_mm_s")


def write_stream(path, t, u, y_temp, y_vib):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for row in zip(t, u, y_temp, y_vib):
            w.writerow(["" if (isinstance(v, float) and math.isnan(v)) else repr(float(v)) for v in row])


def _number(text, decimal_comma):
    text = text.strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    if decimal_comma:
        text = text.replace(",", ".")
    return float(text)


def read_stream(path):
    """Returns ``(t, u, y_temp, y_vib, dt)``; raises ``ValueError`` for missing columns or irregular sampling."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        head = fh.readline()
        fh.seek(0)
        delimiter = ";" if head.count(";") >= head.count(",") and ";" in head else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"veri dosyasında eksik sütun: {', '.join(missing)} (beklenen: {', '.join(COLUMNS)})")
        decimal_comma = delimiter == ";"
        rows = []
        for n, row in enumerate(reader, start=2):
            try:
                rows.append([_number(row[c], decimal_comma) for c in COLUMNS])
            except ValueError:
                raise ValueError(f"{n}. satırda sayı okunamadı: {row}")
    if len(rows) < 10:
        raise ValueError("en az 10 örnek gerekli")
    data = np.array(rows)
    t, u, y_temp, y_vib = data.T
    if np.isnan(t).any() or np.isnan(u).any():
        raise ValueError("zaman ve yük sütunlarında boş değer olamaz")
    steps = np.diff(t)
    dt = float(np.median(steps))
    if dt <= 0 or np.max(np.abs(steps - dt)) > 1e-6 * max(1.0, abs(dt)) + 1e-9:
        raise ValueError("örnekleme aralığı düzenli olmalı (ikiz sabit adımla çalışır)")
    return t, u, y_temp, y_vib, dt
