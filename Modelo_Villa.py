"""
arbol_decision_manual.py
---------------------------------------------------------------------------
Clasificador de actividades humanas mediante un ÁRBOL DE DECISIÓN ÚNICO
(CART, criterio de Gini) IMPLEMENTADO DESDE CERO, es decir, sin usar
sklearn ni ninguna otra biblioteca de aprendizaje de máquina o de
estadística avanzada.

A diferencia de un Bosque Aleatorio (que combina cientos de árboles), aquí
se entrena UN SOLO árbol sobre todo el conjunto de entrenamiento, usando
todas las características disponibles en cada nodo. Esto lo hace mucho más
rápido de entrenar, a cambio de una exactitud algo menor y más riesgo de
sobreajuste, que se controla limitando la profundidad y el mínimo de
muestras por nodo.

Las únicas dependencias externas son numpy (álgebra de arreglos) y pandas
(lectura de archivos tabulares). Ninguna de las dos se usa para entrenar,
predecir o evaluar el modelo: todo eso está programado manualmente en este
archivo (árbol de decisión, métricas, matriz de confusión, etc.). La
búsqueda del mejor corte en cada nodo usa sumas acumuladas vectorizadas de
numpy en vez de un bucle por cada umbral candidato, para que el
entrenamiento sea rápido incluso con miles de filas.

Este script NO depende de un notebook ni de un IDE: se ejecuta completo
desde la terminal con:

    python arbol_decision_manual.py

y opcionalmente acepta la ruta donde están los datos:

    python arbol_decision_manual.py --datos ../datos_modelo
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ===========================================================================
# 1. UTILIDADES DE IMPUREZA Y DIVISIÓN DE NODOS
# ===========================================================================
#
# Un árbol de decisión CART elige, en cada nodo, la característica y el
# umbral que mejor separan las clases. Para medir "qué tan buena" es una
# separación usamos el índice de Gini, calculado a mano.

def indice_gini(etiquetas: np.ndarray, pesos: np.ndarray) -> float:
    """
    Calcula el índice de Gini de un conjunto de etiquetas, ponderando cada
    muestra por 'pesos' (esto es lo que permite simular class_weight
    "balanced" sin usar ninguna función ya implementada).
    """
    peso_total = pesos.sum()
    if peso_total <= 0:
        return 0.0

    gini = 1.0
    for clase in np.unique(etiquetas):
        proporcion = pesos[etiquetas == clase].sum() / peso_total
        gini -= proporcion ** 2
    return gini


def _mejor_umbral_en_columna(
    valores: np.ndarray,
    y_codificado: np.ndarray,
    pesos: np.ndarray,
    n_clases: int,
    gini_padre: float,
) -> tuple[float, float]:
    """
    Encuentra, para UNA columna, el umbral que maximiza la reducción de
    Gini, usando sumas acumuladas vectorizadas con numpy en lugar de
    recorrer cada umbral candidato con un bucle de Python. Esto sigue
    siendo el mismo algoritmo CART hecho a mano (no se usa ninguna
    función de sklearn ni de estadística avanzada), solo se aprovecha
    numpy para hacer la aritmética de golpe.

    Devuelve (mejor_umbral, mejor_ganancia). Si no hay una división
    válida, mejor_umbral es np.nan y mejor_ganancia es 0.0.
    """
    orden = np.argsort(valores, kind="mergesort")
    valores_ordenados = valores[orden]
    y_ordenado = y_codificado[orden]
    pesos_ordenados = pesos[orden]

    peso_total = pesos_ordenados.sum()

    # Matriz (n_muestras, n_clases): peso de la muestra en la columna de
    # su propia clase, 0 en las demás columnas.
    pesos_por_clase = np.zeros((valores_ordenados.shape[0], n_clases))
    pesos_por_clase[np.arange(valores_ordenados.shape[0]), y_ordenado] = pesos_ordenados

    # Acumulados de izquierda a derecha (sumas parciales por clase y peso).
    acumulado_izq_por_clase = np.cumsum(pesos_por_clase, axis=0)
    acumulado_izq_total = np.cumsum(pesos_ordenados)

    acumulado_der_por_clase = acumulado_izq_por_clase[-1] - acumulado_izq_por_clase
    acumulado_der_total = peso_total - acumulado_izq_total

    # Solo son puntos de corte válidos aquellos donde el valor cambia
    # respecto al siguiente, y donde ambos lados quedan no vacíos.
    valores_distintos = np.diff(valores_ordenados) != 0
    lados_no_vacios = (acumulado_izq_total[:-1] > 0) & (acumulado_der_total[:-1] > 0)
    puntos_validos = valores_distintos & lados_no_vacios

    if not puntos_validos.any():
        return float("nan"), 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        gini_izq = 1.0 - np.sum(
            (acumulado_izq_por_clase[:-1] / acumulado_izq_total[:-1, None]) ** 2,
            axis=1,
        )
        gini_der = 1.0 - np.sum(
            (acumulado_der_por_clase[:-1] / acumulado_der_total[:-1, None]) ** 2,
            axis=1,
        )

    gini_hijos = (
        acumulado_izq_total[:-1] * gini_izq + acumulado_der_total[:-1] * gini_der
    ) / peso_total
    ganancias = gini_padre - gini_hijos
    ganancias[~puntos_validos] = -np.inf

    mejor_indice = int(np.argmax(ganancias))
    mejor_ganancia = float(ganancias[mejor_indice])
    if not np.isfinite(mejor_ganancia) or mejor_ganancia <= 0.0:
        return float("nan"), 0.0

    mejor_umbral = (
        valores_ordenados[mejor_indice] + valores_ordenados[mejor_indice + 1]
    ) / 2.0
    return float(mejor_umbral), mejor_ganancia


def mejor_division(
    X: np.ndarray,
    y: np.ndarray,
    pesos: np.ndarray,
    indices_caracteristicas: np.ndarray,
    mapa_clases: dict,
) -> Optional[tuple[int, float, float]]:
    """
    Busca, entre el subconjunto de columnas 'indices_caracteristicas', la
    combinación (columna, umbral) que produce la mayor reducción de
    impureza de Gini al partir el nodo en dos.

    Devuelve (columna, umbral, ganancia) o None si no existe una división
    que mejore la impureza actual.
    """
    n_muestras = X.shape[0]
    gini_padre = indice_gini(y, pesos)
    if gini_padre == 0.0 or n_muestras < 2:
        return None

    y_codificado = np.array([mapa_clases[etiqueta] for etiqueta in y])
    n_clases = len(mapa_clases)

    mejor_ganancia = 0.0
    mejor_columna = None
    mejor_umbral = None

    for columna in indices_caracteristicas:
        umbral, ganancia = _mejor_umbral_en_columna(
            X[:, columna], y_codificado, pesos, n_clases, gini_padre
        )
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_columna = columna
            mejor_umbral = umbral

    if mejor_columna is None:
        return None
    return mejor_columna, mejor_umbral, mejor_ganancia


# ===========================================================================
# 2. ÁRBOL DE DECISIÓN (construido de forma recursiva y manual)
# ===========================================================================

@dataclass
class NodoArbol:
    es_hoja: bool
    prediccion: Optional[str] = None
    columna: Optional[int] = None
    umbral: Optional[float] = None
    izquierdo: Optional["NodoArbol"] = None
    derecho: Optional["NodoArbol"] = None


class ArbolDecisionPropio:
    """
    Árbol de decisión CART simplificado, entrenado con el criterio de Gini.
    Se usa como "aprendiz débil" dentro del bosque aleatorio.
    """

    def __init__(
        self,
        profundidad_maxima: int = 12,
        minimo_muestras_division: int = 2,
        cantidad_caracteristicas_por_nodo: Optional[int] = None,
        semilla: Optional[int] = None,
    ):
        self.profundidad_maxima = profundidad_maxima
        self.minimo_muestras_division = minimo_muestras_division
        self.cantidad_caracteristicas_por_nodo = cantidad_caracteristicas_por_nodo
        self.generador = random.Random(semilla)
        self.raiz: Optional[NodoArbol] = None
        self.total_caracteristicas: int = 0
        self.mapa_clases: dict = {}

    # -- entrenamiento --------------------------------------------------
    def entrenar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pesos: np.ndarray,
        mapa_clases: Optional[dict] = None,
    ) -> None:
        self.total_caracteristicas = X.shape[1]
        self.mapa_clases = mapa_clases if mapa_clases is not None else {
            clase: i for i, clase in enumerate(np.unique(y))
        }
        self.raiz = self._construir_nodo(X, y, pesos, profundidad=0)

    def _construir_nodo(
        self, X: np.ndarray, y: np.ndarray, pesos: np.ndarray, profundidad: int
    ) -> NodoArbol:
        clase_mayoritaria = self._clase_mas_pesada(y, pesos)

        condiciones_de_parada = (
            profundidad >= self.profundidad_maxima
            or X.shape[0] < self.minimo_muestras_division
            or np.unique(y).size == 1
        )
        if condiciones_de_parada:
            return NodoArbol(es_hoja=True, prediccion=clase_mayoritaria)

        columnas_candidatas = self._elegir_subconjunto_columnas()
        resultado = mejor_division(X, y, pesos, columnas_candidatas, self.mapa_clases)

        if resultado is None:
            return NodoArbol(es_hoja=True, prediccion=clase_mayoritaria)

        columna, umbral, _ganancia = resultado
        mascara_izq = X[:, columna] <= umbral

        hijo_izquierdo = self._construir_nodo(
            X[mascara_izq], y[mascara_izq], pesos[mascara_izq], profundidad + 1
        )
        hijo_derecho = self._construir_nodo(
            X[~mascara_izq], y[~mascara_izq], pesos[~mascara_izq], profundidad + 1
        )

        return NodoArbol(
            es_hoja=False,
            columna=columna,
            umbral=umbral,
            izquierdo=hijo_izquierdo,
            derecho=hijo_derecho,
        )

    def _elegir_subconjunto_columnas(self) -> np.ndarray:
        if self.cantidad_caracteristicas_por_nodo is None:
            return np.arange(self.total_caracteristicas)

        cantidad = min(
            self.cantidad_caracteristicas_por_nodo, self.total_caracteristicas
        )
        columnas = self.generador.sample(range(self.total_caracteristicas), cantidad)
        return np.array(columnas)

    @staticmethod
    def _clase_mas_pesada(y: np.ndarray, pesos: np.ndarray) -> str:
        peso_por_clase: dict[str, float] = {}
        for clase, peso in zip(y, pesos):
            peso_por_clase[clase] = peso_por_clase.get(clase, 0.0) + peso
        return max(peso_por_clase, key=peso_por_clase.get)

    # -- predicción -------------------------------------------------------
    def predecir_una(self, fila: np.ndarray) -> str:
        nodo = self.raiz
        while not nodo.es_hoja:
            if fila[nodo.columna] <= nodo.umbral:
                nodo = nodo.izquierdo
            else:
                nodo = nodo.derecho
        return nodo.prediccion

    def predecir(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predecir_una(fila) for fila in X])


# ===========================================================================
# 3. ENTRENADOR DEL ÁRBOL ÚNICO (sin bagging, sin ensamble)
# ===========================================================================
#
# A diferencia del Bosque Aleatorio, aquí NO se hace muestreo bootstrap ni
# se sortean subconjuntos de características: el árbol ve todas las filas
# y todas las columnas del conjunto de entrenamiento, como un árbol de
# decisión clásico (CART). El único "truco" para que sea más robusto es
# ponderar las muestras por clase (balanceo), igual que antes.

@dataclass
class ClasificadorArbolUnico:
    profundidad_maxima: int = 20
    minimo_muestras_division: int = 2
    balancear_clases: bool = True
    semilla: int = 42
    arbol: Optional[ArbolDecisionPropio] = None

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        n_muestras = X.shape[0]
        mapa_clases = {clase: i for i, clase in enumerate(np.unique(y))}

        pesos = self._calcular_pesos_balanceados(y) if self.balancear_clases \
            else np.ones(n_muestras)

        inicio = time.time()
        self.arbol = ArbolDecisionPropio(
            profundidad_maxima=self.profundidad_maxima,
            minimo_muestras_division=self.minimo_muestras_division,
            cantidad_caracteristicas_por_nodo=None,  # usa TODAS las columnas
            semilla=self.semilla,
        )
        self.arbol.entrenar(X, y, pesos, mapa_clases)
        print(f"      Árbol entrenado en {time.time() - inicio:.2f}s")

    @staticmethod
    def _calcular_pesos_balanceados(y: np.ndarray) -> np.ndarray:
        """
        Reproduce manualmente el efecto de class_weight="balanced":
        peso_clase = n_muestras / (n_clases * frecuencia_de_la_clase)
        """
        n_muestras = y.shape[0]
        clases, conteos = np.unique(y, return_counts=True)
        n_clases = clases.shape[0]

        peso_por_clase = {
            clase: n_muestras / (n_clases * conteo)
            for clase, conteo in zip(clases, conteos)
        }
        return np.array([peso_por_clase[etiqueta] for etiqueta in y])

    def predecir(self, X: np.ndarray) -> np.ndarray:
        return self.arbol.predecir(X)


# ===========================================================================
# 4. MÉTRICAS DE EVALUACIÓN (accuracy, precision, recall, F1, matriz)
#    -> también programadas a mano, sin sklearn.metrics
# ===========================================================================

def exactitud_manual(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_real == y_pred))


def matriz_confusion_manual(y_real: np.ndarray, y_pred: np.ndarray, clases) -> np.ndarray:
    indice = {clase: i for i, clase in enumerate(clases)}
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    for real, pred in zip(y_real, y_pred):
        matriz[indice[real], indice[pred]] += 1
    return matriz


def reporte_clasificacion_manual(y_real: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    clases = sorted(np.unique(np.concatenate([y_real, y_pred])))
    filas = []

    for clase in clases:
        verdaderos_positivos = int(np.sum((y_pred == clase) & (y_real == clase)))
        falsos_positivos = int(np.sum((y_pred == clase) & (y_real != clase)))
        falsos_negativos = int(np.sum((y_pred != clase) & (y_real == clase)))
        soporte = int(np.sum(y_real == clase))

        precision = (
            verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
            if (verdaderos_positivos + falsos_positivos) > 0 else 0.0
        )
        recall = (
            verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
            if (verdaderos_positivos + falsos_negativos) > 0 else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        filas.append({
            "actividad": clase,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "soporte": soporte,
        })

    tabla = pd.DataFrame(filas)

    promedio_macro = tabla[["precision", "recall", "f1_score"]].mean()
    soporte_total = tabla["soporte"].sum()
    promedio_ponderado = (
        tabla[["precision", "recall", "f1_score"]]
        .mul(tabla["soporte"], axis=0)
        .sum() / soporte_total
    )

    tabla.loc[len(tabla)] = {
        "actividad": "promedio_macro",
        "precision": round(promedio_macro["precision"], 4),
        "recall": round(promedio_macro["recall"], 4),
        "f1_score": round(promedio_macro["f1_score"], 4),
        "soporte": soporte_total,
    }
    tabla.loc[len(tabla)] = {
        "actividad": "promedio_ponderado",
        "precision": round(promedio_ponderado["precision"], 4),
        "recall": round(promedio_ponderado["recall"], 4),
        "f1_score": round(promedio_ponderado["f1_score"], 4),
        "soporte": soporte_total,
    }
    return tabla


# ===========================================================================
# 5. CARGA DE DATOS Y AGREGACIÓN POR SEÑAL COMPLETA
# ===========================================================================

def cargar_datos(ruta_datos: Path):
    X_train = np.load(ruta_datos / "X_train.npy", allow_pickle=False)
    X_test = np.load(ruta_datos / "X_test.npy", allow_pickle=False)
    y_train = np.load(ruta_datos / "y_train.npy", allow_pickle=False)
    y_test = np.load(ruta_datos / "y_test.npy", allow_pickle=False)
    nombres_caracteristicas = np.load(
        ruta_datos / "nombres_caracteristicas.npy", allow_pickle=True
    )
    return X_train, X_test, y_train, y_test, nombres_caracteristicas


def agregar_prediccion_por_senal(metadatos_test: pd.DataFrame) -> pd.DataFrame:
    """
    Combina las 4 ventanas de cada señal en una sola predicción, usando
    la moda de las predicciones (votación mayoritaria), igual que el
    criterio original del proyecto pero calculado sin funciones de
    estadística de sklearn/scipy.
    """
    def moda(serie: pd.Series):
        return Counter(serie).most_common(1)[0][0]

    resultados = (
        metadatos_test
        .groupby("sample_id", as_index=False)
        .agg(
            actividad_real=("actividad_real", "first"),
            actividad_predicha=("actividad_predicha", moda),
            cantidad_ventanas=("numero_ventana", "count"),
        )
    )
    return resultados


# ===========================================================================
# 6. PROGRAMA PRINCIPAL
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Árbol de decisión único implementado desde cero (sin frameworks de ML)."
    )
    parser.add_argument(
        "--datos",
        type=str,
        default="../datos_modelo",
        help="Carpeta con X_train.npy, X_test.npy, y_train.npy, y_test.npy, "
             "nombres_caracteristicas.npy y metadatos_test.csv",
    )
    parser.add_argument("--profundidad", type=int, default=20, help="Profundidad máxima del árbol")
    parser.add_argument(
        "--min-muestras", type=int, default=2,
        help="Mínimo de muestras para seguir dividiendo un nodo",
    )
    parser.add_argument("--semilla", type=int, default=42, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    ruta_datos = Path(args.datos)

    print("=" * 70)
    print("ÁRBOL DE DECISIÓN MANUAL — sin sklearn ni frameworks de ML")
    print("=" * 70)

    print(f"\n[1/5] Cargando datos desde: {ruta_datos.resolve()}")
    X_train, X_test, y_train, y_test, nombres_caracteristicas = cargar_datos(ruta_datos)
    print(f"      X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"      X_test:  {X_test.shape}    y_test:  {y_test.shape}")
    print(f"      Características: {nombres_caracteristicas.shape[0]}")

    print(f"\n[2/5] Entrenando árbol de decisión propio "
          f"(profundidad máx. {args.profundidad}, mín. muestras por nodo {args.min_muestras})...")
    clasificador = ClasificadorArbolUnico(
        profundidad_maxima=args.profundidad,
        minimo_muestras_division=args.min_muestras,
        balancear_clases=True,
        semilla=args.semilla,
    )
    clasificador.entrenar(X_train, y_train)

    print("\n[3/5] Generando predicciones sobre el conjunto de prueba...")
    y_pred = clasificador.predecir(X_test)

    exactitud = exactitud_manual(y_test, y_pred)
    print(f"      Exactitud por ventana: {exactitud:.4f} ({exactitud * 100:.2f}%)")

    print("\n      Reporte de clasificación por ventana:\n")
    print(reporte_clasificacion_manual(y_test, y_pred).to_string(index=False))

    clases_ordenadas = sorted(np.unique(np.concatenate([y_test, y_pred])))
    matriz = matriz_confusion_manual(y_test, y_pred, clases_ordenadas)
    print("\n      Matriz de confusión (filas=real, columnas=predicho):")
    print(pd.DataFrame(matriz, index=clases_ordenadas, columns=clases_ordenadas))

    ruta_metadatos = ruta_datos / "metadatos_test.csv"
    if ruta_metadatos.exists():
        print("\n[4/5] Agregando predicciones por señal completa (votación de 4 ventanas)...")
        metadatos_test = pd.read_csv(
            ruta_metadatos, dtype={"sample_id": str, "actividad": str}
        )
        metadatos_test["actividad_real"] = y_test
        metadatos_test["actividad_predicha"] = y_pred

        resultados_senales = agregar_prediccion_por_senal(metadatos_test)

        exactitud_senal = exactitud_manual(
            resultados_senales["actividad_real"].to_numpy(),
            resultados_senales["actividad_predicha"].to_numpy(),
        )
        print(f"      Señales evaluadas: {len(resultados_senales)}")
        print(f"      Exactitud por señal completa: "
              f"{exactitud_senal:.4f} ({exactitud_senal * 100:.2f}%)")
    else:
        print("\n[4/5] No se encontró metadatos_test.csv; se omite la agregación por señal.")

    print("\n[5/5] Proceso terminado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
