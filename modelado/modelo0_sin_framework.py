"""
Modelo KNN implementado desde cero para clasificar actividades REHAB.

Este programa puede ejecutarse directamente desde una terminal y no depende
de Jupyter Notebook ni de un IDE. NumPy se utiliza para leer los datos y
realizar operaciones matematicas, pero el algoritmo KNN, la votacion y el
calculo de la exactitud estan implementados manualmente.

Ejecucion:
    python3 modelo0_sin_framework.py
"""

from csv import DictReader
from pathlib import Path

import numpy as np


# La ruta se calcula desde la ubicacion de este archivo, por lo que el programa
# funciona aunque se ejecute desde otra carpeta.
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS_MODELO = CARPETA_PROYECTO / "datos_modelo"


def cargar_datos():
    """Carga los cuatro arreglos preparados y los metadatos de prueba."""

    X_train = np.load(
        RUTA_DATOS_MODELO / "X_train.npy", allow_pickle=False
    )
    X_test = np.load(
        RUTA_DATOS_MODELO / "X_test.npy", allow_pickle=False
    )
    y_train = np.load(
        RUTA_DATOS_MODELO / "y_train.npy", allow_pickle=False
    )
    y_test = np.load(
        RUTA_DATOS_MODELO / "y_test.npy", allow_pickle=False
    )

    # Solo se necesita sample_id para reunir las cuatro ventanas que proceden
    # de una misma señal. Este identificador no se usa como entrada del KNN.
    ruta_metadatos = RUTA_DATOS_MODELO / "metadatos_test.csv"
    with ruta_metadatos.open(encoding="utf-8", newline="") as archivo:
        sample_ids = np.asarray(
            [fila["sample_id"] for fila in DictReader(archivo)]
        )

    if len(X_train) != len(y_train):
        raise ValueError("X_train y y_train no tienen la misma longitud.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test y y_test no tienen la misma longitud.")
    if len(X_test) != len(sample_ids):
        raise ValueError("Los metadatos no coinciden con las ventanas de prueba.")

    return X_train, X_test, y_train, y_test, sample_ids


def estandarizar_datos(X_train, X_test):
    """Estandariza ambos conjuntos usando solamente parametros de train."""

    media_train = np.mean(X_train, axis=0)
    desviacion_train = np.std(X_train, axis=0)

    # Evita divisiones entre cero en caracteristicas constantes.
    desviacion_train[desviacion_train == 0] = 1

    X_train_escalado = (X_train - media_train) / desviacion_train
    X_test_escalado = (X_test - media_train) / desviacion_train

    # Se conserva float64 para evitar desbordamientos numericos en las
    # multiplicaciones utilizadas para calcular las distancias.
    return (
        X_train_escalado.astype(np.float64),
        X_test_escalado.astype(np.float64),
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
        """Calcula vecinos y predicciones para un bloque de ventanas."""

        # Distancia euclidiana al cuadrado:
        # ||a-b||² = ||a||² + ||b||² - 2(a·b)
        norma_prueba = np.sum(X_bloque**2, axis=1, keepdims=True)
        # Algunas versiones de la biblioteca numerica de macOS pueden emitir
        # advertencias internas espurias durante matmul, aun cuando el
        # resultado es finito. El control siguiente evita mostrarlas.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            distancias = (
                norma_prueba
                + self.norma_train
                - 2 * X_bloque @ self.X_train.T
            )
        distancias = np.maximum(distancias, 0)

        # Obtiene los indices de los k vecinos mas cercanos sin ordenar toda
        # la matriz de distancias.
        indices_vecinos = np.argpartition(
            distancias, kth=self.k - 1, axis=1
        )[:, : self.k]

        predicciones = []
        for fila, vecinos in enumerate(indices_vecinos):
            etiquetas_vecinos = self.y_numerico[vecinos]
            votos = np.bincount(
                etiquetas_vecinos, minlength=len(self.clases)
            )
            ganadoras = np.flatnonzero(votos == votos.max())

            if len(ganadoras) == 1:
                clase_elegida = ganadoras[0]
            else:
                # Si el voto del KNN empata, gana la clase empatada cuyo
                # vecino se encuentre mas cerca.
                orden = np.argsort(distancias[fila, vecinos])
                clase_elegida = next(
                    etiquetas_vecinos[posicion]
                    for posicion in orden
                    if etiquetas_vecinos[posicion] in ganadoras
                )

            predicciones.append(self.clases[clase_elegida])

        return np.asarray(predicciones)

    def predict(self, X):
        """Predice las etiquetas procesando las ventanas por bloques."""

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


def votar_por_senal(sample_ids, y_real, y_predicha):
    """Combina las cuatro predicciones mediante votacion mayoritaria."""

    grupos = {}
    for sample_id, real, predicha in zip(sample_ids, y_real, y_predicha):
        if sample_id not in grupos:
            grupos[sample_id] = {"real": real, "predicciones": []}
        grupos[sample_id]["predicciones"].append(predicha)

    reales_senal = []
    predicciones_senal = []

    for grupo in grupos.values():
        conteo = {}
        for prediccion in grupo["predicciones"]:
            conteo[prediccion] = conteo.get(prediccion, 0) + 1

        # max conserva la primera clase encontrada cuando existe un empate.
        prediccion_final = max(conteo, key=conteo.get)
        reales_senal.append(grupo["real"])
        predicciones_senal.append(prediccion_final)

    return np.asarray(reales_senal), np.asarray(predicciones_senal)


def main():
    """Ejecuta de principio a fin la prueba del modelo manual."""

    print("Cargando datos preparados...")
    X_train, X_test, y_train, y_test, sample_ids = cargar_datos()

    print("Estandarizando caracteristicas...")
    X_train_escalado, X_test_escalado = estandarizar_datos(
        X_train, X_test
    )

    print("Entrenando KNN manual con k=5...")
    modelo = KNNDesdeCero(k=5, tamanio_bloque=64)
    modelo.fit(X_train_escalado, y_train)

    print("Realizando predicciones...")
    y_pred_knn = modelo.predict(X_test_escalado)

    # Muestra algunas predicciones solicitadas por la actividad.
    print("\nPrimeras 10 predicciones por ventana:")
    for posicion in range(min(10, len(y_test))):
        print(
            f"Ventana {posicion + 1}: real={y_test[posicion]}, "
            f"predicha={y_pred_knn[posicion]}"
        )

    aciertos_ventana, exactitud_ventana = calcular_exactitud(
        y_test, y_pred_knn
    )
    print("\nEvaluacion por ventana")
    print(f"Aciertos: {aciertos_ventana} de {len(y_test)}")
    print(f"Exactitud: {exactitud_ventana * 100:.2f}%")

    y_real_senal, y_pred_senal = votar_por_senal(
        sample_ids, y_test, y_pred_knn
    )
    aciertos_senal, exactitud_senal = calcular_exactitud(
        y_real_senal, y_pred_senal
    )
    print("\nEvaluacion por señal completa")
    print(f"Aciertos: {aciertos_senal} de {len(y_real_senal)}")
    print(f"Exactitud: {exactitud_senal * 100:.2f}%")


if __name__ == "__main__":
    main()
