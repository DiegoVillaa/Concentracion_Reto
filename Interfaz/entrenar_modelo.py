"""Entrena y guarda el Random Forest definitivo utilizado por la interfaz."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model_utils import feature_names, project_root


ROOT = project_root()
DATA_DIR = ROOT / "Datos modelo"
OUTPUT = Path(__file__).resolve().parent / "modelo_random_forest.joblib"


def main() -> None:
    train = pd.read_csv(DATA_DIR / "datos_train.csv", dtype={"actividad": str})
    validation = pd.read_csv(DATA_DIR / "datos_validation.csv", dtype={"actividad": str})
    development = pd.concat([train, validation], ignore_index=True)

    expected_features = feature_names()
    missing = [column for column in expected_features if column not in development.columns]
    if missing:
        raise ValueError(f"Faltan características esperadas: {missing[:5]}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(development[expected_features], development["actividad"])

    artifact = {
        "model": model,
        "feature_names": expected_features,
        "metadata": {
            "algorithm": "Random Forest",
            "training_rows": len(development),
            "window_size": 220,
            "channels": 12,
            "test_accuracy": 0.9563,
            "test_f1_macro": 0.9531,
        },
    }
    joblib.dump(artifact, OUTPUT, compress=3)
    print(f"Modelo guardado en: {OUTPUT}")
    print(f"Filas utilizadas (train + validation): {len(development)}")


if __name__ == "__main__":
    main()
