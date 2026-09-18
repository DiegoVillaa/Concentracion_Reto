# Interfaz web — REHAB Motion Lab

Esta aplicación utiliza el Random Forest definitivo para clasificar las cuatro ventanas de una señal cinemática de rehabilitación. Permite seleccionar un ejemplo local del dataset o cargar los dos archivos `.npy` correspondientes a los grupos de sensores 1 y 2. También muestra la configuración final seleccionada mediante Grid Search manual y las métricas obtenidas en test.

## Ejecución

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Interfaz/requirements.txt
python -m streamlit run Interfaz/app.py
```

El modelo entrenado ya se encuentra en `Interfaz/modelo_random_forest.joblib`. Si fuera necesario reconstruirlo, se puede ejecutar `python Interfaz/entrenar_modelo.py`; el script utiliza `datos_train.csv` y `datos_validation.csv`, sin incorporar el conjunto de test al entrenamiento.

## Entrada esperada

Cada archivo puede contener una repetición con forma `(880, 6)` o un lote con forma `(n, 880, 6)`. Los dos archivos deben pertenecer a la misma actividad y contener la misma cantidad de repeticiones.

## Interpretación

La interfaz presenta una predicción para cada ventana de 220 puntos. La “actividad dominante” es solamente un resumen visual de las cuatro predicciones; no se utilizó votación para calcular las métricas del proyecto.
