"""Modelo KNN implementado desde cero para clasificar actividades REHAB.

Se ejecuta desde terminal y lee datos_train.csv y datos_test.csv. Cada fila
es una observacion independiente con 120 caracteristicas y una actividad.
"""

from pathlib import Path

import numpy as np
import pandas as pd


CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS_MODELO = CARPETA_PROYECTO / "datos_modelo"
 

def cargar_datos():
    """Carga los CSV y separa las caracteristicas de la etiqueta."""

    datos_train = pd.read_csv(
        RUTA_DATOS_MODELO / "datos_train.csv",
        dtype={"actividad": str},
    )
    datos_test = pd.read_csv(
        RUTA_DATOS_MODELO / "datos_test.csv",
        dtype={"actividad": str},
    )

    if datos_train.columns[-1] != "actividad":
        raise ValueError("actividad debe ser la ultima columna de train.")
    if datos_test.columns[-1] != "actividad":
        raise ValueError("actividad debe ser la ultima columna de test.")

    X_train = datos_train.drop(columns="actividad").to_numpy()
    X_test = datos_test.drop(columns="actividad").to_numpy()
    y_train = datos_train["actividad"].to_numpy()
    y_test = datos_test["actividad"].to_numpy()

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Las caracteristicas y etiquetas no estan alineadas.")

    return X_train, X_test, y_train, y_test


def estandarizar_datos(X_train, X_test):
    """Estandariza usando solamente la media y desviacion de train."""

    media_train = np.mean(X_train, axis=0)
    desviacion_train = np.std(X_train, axis=0)
    desviacion_train[desviacion_train == 0] = 1

    return (
        ((X_train - media_train) / desviacion_train).astype(np.float64),
        ((X_test - media_train) / desviacion_train).astype(np.float64),
    )


class KNNDesdeCero:
    """Clasificador K-Nearest Neighbors implementado con NumPy."""

    def __init__(self, k=5, tamanio_bloque=64):
        self.k = k
        self.tamanio_bloque = tamanio_bloque

    def fit(self, X, y):
        """Guarda las observaciones y etiquetas de entrenamiento."""

        if len(X) != len(y):
            raise ValueError("X y y deben tener la misma cantidad de filas.")
        if self.k < 1 or self.k > len(X):
            raise ValueError("k debe estar entre 1 y la cantidad de muestras.")

        self.X_train = np.asarray(X, dtype=np.float64)
        self.clases, self.y_numerico = np.unique(y, return_inverse=True)
        self.norma_train = np.sum(self.X_train**2, axis=1)
        return self

    def _predecir_bloque(self, X_bloque):
        """Calcula vecinos y predicciones para un bloque de datos."""

        # Distancia euclidiana al cuadrado:
        # ||a-b||² = ||a||² + ||b||² - 2(a·b)
        norma_prueba = np.sum(X_bloque**2, axis=1, keepdims=True)

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            distancias = (
                norma_prueba
                + self.norma_train
                - 2 * X_bloque @ self.X_train.T
            )

        distancias = np.maximum(distancias, 0)
        indices_vecinos = np.argpartition(
            distancias, kth=self.k - 1, axis=1
        )[:, : self.k]

        predicciones = []

        for fila, vecinos in enumerate(indices_vecinos):
            etiquetas_vecinos = self.y_numerico[vecinos]
            votos = np.bincount(
                etiquetas_vecinos, minlength=len(self.clases)
            )
            clases_ganadoras = np.flatnonzero(votos == votos.max())

            if len(clases_ganadoras) == 1:
                clase_elegida = clases_ganadoras[0]
            else:
                # En un empate gana la clase empatada cuyo vecino este
                # mas cerca de la observacion evaluada.
                orden = np.argsort(distancias[fila, vecinos])
                clase_elegida = next(
                    etiquetas_vecinos[posicion]
                    for posicion in orden
                    if etiquetas_vecinos[posicion] in clases_ganadoras
                )

            predicciones.append(self.clases[clase_elegida])

        return np.asarray(predicciones)

    def predict(self, X):
        """Predice las etiquetas procesando las observaciones por bloques."""

        X = np.asarray(X, dtype=np.float64)
        predicciones = []

        for inicio in range(0, len(X), self.tamanio_bloque):
            fin = inicio + self.tamanio_bloque
            predicciones.extend(self._predecir_bloque(X[inicio:fin]))

        return np.asarray(predicciones)


def calcular_exactitud(y_real, y_predicha):
    """Calcula manualmente la proporcion de predicciones correctas."""

    aciertos = int(np.sum(y_real == y_predicha))
    return aciertos, aciertos / len(y_real)


def calcular_metricas_clasificacion(y_real, y_predicha):
    """Calcula manualmente matriz, precision, recall y F1 por actividad."""

    clases = np.unique(np.concatenate([y_real, y_predicha]))
    posicion_clase = {
        clase: posicion for posicion, clase in enumerate(clases)
    }

    # Las filas representan clases reales y las columnas clases predichas.
    matriz = np.zeros((len(clases), len(clases)), dtype=int)

    for real, predicha in zip(y_real, y_predicha):
        matriz[posicion_clase[real], posicion_clase[predicha]] += 1

    verdaderos_positivos = np.diag(matriz).astype(float)
    falsos_positivos = matriz.sum(axis=0) - verdaderos_positivos
    falsos_negativos = matriz.sum(axis=1) - verdaderos_positivos
    soporte = matriz.sum(axis=1)

    precision = np.divide(
        verdaderos_positivos,
        verdaderos_positivos + falsos_positivos,
        out=np.zeros_like(verdaderos_positivos),
        where=(verdaderos_positivos + falsos_positivos) != 0,
    )
    recall = np.divide(
        verdaderos_positivos,
        verdaderos_positivos + falsos_negativos,
        out=np.zeros_like(verdaderos_positivos),
        where=(verdaderos_positivos + falsos_negativos) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    reporte = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": soporte,
        },
        index=clases,
    )

    promedios = {
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "precision_ponderada": float(np.average(precision, weights=soporte)),
        "recall_ponderado": float(np.average(recall, weights=soporte)),
        "f1_ponderado": float(np.average(f1, weights=soporte)),
    }

    return clases, matriz, reporte, promedios


def main():
    """Carga los datos, entrena KNN y muestra algunas predicciones."""

    print("Cargando datos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = cargar_datos()
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")

    print("Estandarizando caracteristicas...")
    X_train_escalado, X_test_escalado = estandarizar_datos(
        X_train, X_test
    )

    print("Entrenando KNN manual con k=5...")
    modelo = KNNDesdeCero(k=5, tamanio_bloque=64)
    modelo.fit(X_train_escalado, y_train)

    print("Realizando predicciones...")
    y_pred_knn = modelo.predict(X_test_escalado)

    print("\nPrimeras 10 predicciones:")
    for posicion in range(min(10, len(y_test))):
        print(
            f"Observacion {posicion + 1}: real={y_test[posicion]}, "
            f"predicha={y_pred_knn[posicion]}"
        )

    aciertos, exactitud = calcular_exactitud(y_test, y_pred_knn)
    print("\nEvaluacion del modelo")
    print(f"Predicciones correctas: {aciertos} de {len(y_test)}")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Porcentaje de aciertos: {exactitud * 100:.2f}%")

    clases, matriz, reporte, promedios = calcular_metricas_clasificacion(
        y_test, y_pred_knn
    )

    print("\nMetricas por actividad")
    print(reporte.round(4).to_string())

    print("\nPromedios generales")
    print(f"Precision macro: {promedios['precision_macro']:.4f}")
    print(f"Recall macro: {promedios['recall_macro']:.4f}")
    print(f"F1-score macro: {promedios['f1_macro']:.4f}")
    print(
        "F1-score ponderado: "
        f"{promedios['f1_ponderado']:.4f}"
    )

    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real {clase}" for clase in clases],
        columns=[f"Pred {clase}" for clase in clases],
    )

    print("\nMatriz de confusion")
    print(matriz_df.to_string())


if __name__ == "__main__":
    main()
