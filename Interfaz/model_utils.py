"""Funciones compartidas por el entrenamiento y la interfaz web."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np


CHANNEL_NAMES = [
    "pitch1", "yaw1", "roll1", "pitch2", "yaw2", "roll2",
    "f1", "f2", "f3", "f4", "f5", "pitch3",
]

STAT_NAMES = [
    "media", "mediana", "desviacion", "minimo", "maximo", "rango",
    "percentil_25", "percentil_75", "rango_intercuartil", "rms",
]

WINDOW_SIZE = 220
POINTS_PER_SIGNAL = 880


def feature_names() -> list[str]:
    """Devuelve los nombres en el mismo orden utilizado por el ETL."""
    return [f"{channel}_{stat}" for stat in STAT_NAMES for channel in CHANNEL_NAMES]


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Resume cada ventana de 220 x 12 en 120 características estadísticas."""
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SIZE, len(CHANNEL_NAMES)):
        raise ValueError("Las ventanas deben tener dimensiones (n, 220, 12).")
    if not np.isfinite(windows).all():
        raise ValueError("La señal contiene valores faltantes o infinitos.")

    minimum = np.min(windows, axis=1)
    maximum = np.max(windows, axis=1)
    percentile_25 = np.percentile(windows, 25, axis=1)
    percentile_75 = np.percentile(windows, 75, axis=1)

    return np.concatenate(
        [
            np.mean(windows, axis=1),
            np.median(windows, axis=1),
            np.std(windows, axis=1),
            minimum,
            maximum,
            maximum - minimum,
            percentile_25,
            percentile_75,
            percentile_75 - percentile_25,
            np.sqrt(np.mean(windows**2, axis=1)),
        ],
        axis=1,
    )


def split_into_windows(signal: np.ndarray) -> np.ndarray:
    """Divide una señal completa de 880 x 12 en cuatro ventanas consecutivas."""
    signal = np.asarray(signal, dtype=float)
    if signal.shape != (POINTS_PER_SIGNAL, len(CHANNEL_NAMES)):
        raise ValueError("La señal combinada debe tener dimensiones (880, 12).")
    if not np.isfinite(signal).all():
        raise ValueError("La señal contiene valores faltantes o infinitos.")
    return signal.reshape(POINTS_PER_SIGNAL // WINDOW_SIZE, WINDOW_SIZE, len(CHANNEL_NAMES))


def combine_sensor_groups(group_1: np.ndarray, group_2: np.ndarray) -> np.ndarray:
    """Une las seis columnas de cada grupo para producir una señal de 12 canales."""
    group_1 = np.asarray(group_1, dtype=float)
    group_2 = np.asarray(group_2, dtype=float)
    if group_1.shape != (POINTS_PER_SIGNAL, 6) or group_2.shape != (POINTS_PER_SIGNAL, 6):
        raise ValueError("Cada grupo debe representar una repetición con dimensiones (880, 6).")
    return np.concatenate([group_1, group_2], axis=1)


def load_uploaded_npy(uploaded_file) -> np.ndarray:
    """Carga de forma segura un archivo .npy recibido por Streamlit."""
    return np.load(BytesIO(uploaded_file.getvalue()), allow_pickle=False)


def select_repetition(array: np.ndarray, index: int) -> np.ndarray:
    """Obtiene una repetición de un archivo individual o de un lote."""
    if array.ndim == 2:
        if index != 0:
            raise ValueError("El archivo contiene una sola repetición.")
        return array
    if array.ndim == 3:
        return array[index]
    raise ValueError("El archivo debe tener dimensiones (880, 6) o (n, 880, 6).")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
