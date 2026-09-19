# Reto REHAB

Este repositorio contiene el desarrollo de la solución para el reto REHAB de la concentración **Inteligencia Artificial para la Ciencia de Datos**.

## Documentación final

Los entregables finales del proyecto se encuentran en la carpeta [`Documentación`](./Documentaci%C3%B3n/). Para facilitar su revisión, pueden consultarse directamente desde los siguientes enlaces:

- **[Reporte final del reto REHAB](./Documentaci%C3%B3n/Reporte_final_REHAB.pdf):** documenta el problema, análisis exploratorio, reconstrucción del archivo corrupto, ETL, modelación, refinamiento, resultados, interfaz y conclusiones.
- **[Presentación final del reto REHAB](./Documentaci%C3%B3n/Reto%201%20Presentaci%C3%B3n.pdf):** resume el enfoque seguido, las decisiones principales y los resultados obtenidos.

Se recomienda comenzar por la presentación para obtener una visión general y consultar después el reporte para revisar el procedimiento completo y su justificación.

## Introducción al reto

Un accidente cerebrovascular puede afectar la movilidad, la fuerza y la coordinación de una persona, por lo que la evaluación y el entrenamiento de rehabilitación son importantes durante su recuperación. El conjunto de datos REHAB contiene registros cinemáticos obtenidos mediante sensores portátiles durante distintos movimientos de rehabilitación.

Para este proyecto se utilizan los datos procesados correspondientes a **16 actividades de entrenamiento**, identificadas con las etiquetas `000` a `015`. El objetivo es preparar las señales y construir un modelo de clasificación capaz de identificar la actividad realizada a partir de las mediciones de los sensores.

## Flujo general del proyecto

El trabajo principal sigue este orden:

1. Exploración inicial de los archivos y las señales.
2. Reconstrucción y validación del archivo procesado `014_1.npy`.
3. Limpieza y preparación de los datos mediante el ETL definitivo.
4. Separación de las señales completas en entrenamiento, validación y prueba.
5. División de cada señal en ventanas y extracción de características estadísticas.
6. Entrenamiento y comparación de tres modelos de clasificación.
7. Revisión de sobreajuste, optimización de Random Forest y evaluación detallada.

## Estructura del repositorio

```text
Concentracion_Reto/
├── Rehab_exercise/
│   ├── d01_raw_data/              # Archivos raw usados en la reconstrucción
│   └── d02_processed_data/        # 32 archivos procesados de las 16 actividades
├── Exploración inicial/           # Notebooks del EDA
├── Reconstrucción de archivo corrupto/
│   ├── reconstruir_014_1.py
│   └── comparar_reconstruidos.ipynb
├── ETL/
│   └── etl_definitivo.ipynb       # Preparación acordada por el equipo
├── Datos modelo/
│   ├── datos_train.csv
│   ├── datos_validation.csv
│   └── datos_test.csv
├── modelado/
│   └── modelacion_3_modelos_def.ipynb
├── Interfaz/
│   ├── app.py                     # Aplicación web en Streamlit
│   ├── modelo_random_forest.joblib
│   ├── model_utils.py             # Ventanas y extracción de características
│   └── entrenar_modelo.py         # Regeneración opcional del modelo
├── Documentación/
│   ├── Reporte_final_REHAB.pdf     # Reporte completo del proyecto
│   └── Reto 1 Presentación.pdf     # Presentación final del equipo
└── Momento - Redefinición de datos/  # Propuestas y pruebas anteriores
```

## Reconstrucción de `014_1.npy`

El archivo procesado original `014_1.npy` presentó corrupción binaria. Se reconstruyó a partir de su versión raw aplicando el procedimiento de filtrado y normalización descrito por los autores del conjunto de datos.

La transformación se verificó con seis archivos oficiales. Se conservaron sus dimensiones y se obtuvo igualdad numérica dentro de la tolerancia de punto flotante. El archivo reconstruido se encuentra actualmente en `Rehab_exercise/d02_processed_data`, por lo que el ETL puede cargar directamente los 32 archivos procesados.

Los detalles y el código de validación se conservan en `Reconstrucción de archivo corrupto`.

## ETL definitivo

Después de revisar y discutir las propuestas individuales, el equipo definió el procedimiento ubicado en `ETL/etl_definitivo.ipynb`.

El ETL realiza las siguientes operaciones:

- Carga y valida los 32 archivos `.npy` procesados.
- Une los dos grupos de sensores de cada actividad para obtener señales de `880 × 12`.
- Asigna la etiqueta de actividad correspondiente.
- Elimina 68 señales completamente en cero.
- Detecta duplicados exactos considerando los 880 puntos y los 12 canales.
- Elimina 658 copias adicionales y conserva una señal representativa por grupo.
- Conserva 3,890 señales limpias de las 16 actividades.
- Separa las **señales completas** en 60% entrenamiento, 20% validación y 20% prueba mediante estratificación.
- Divide cada señal en cuatro ventanas consecutivas de 220 puntos, sin traslape.
- Calcula diez estadísticas para cada uno de los 12 canales: media, mediana, desviación estándar, mínimo, máximo, rango, percentiles 25 y 75, rango intercuartílico y RMS.
- Genera 120 características por ventana y guarda las tablas finales.

La separación se realiza antes de crear las ventanas. Por lo tanto, las cuatro ventanas procedentes de una señal permanecen juntas en entrenamiento, validación o prueba, evitando que fragmentos relacionados aparezcan en conjuntos diferentes.

Los archivos generados son:

- `Datos modelo/datos_train.csv`: 9,336 ventanas, 120 características y la etiqueta.
- `Datos modelo/datos_validation.csv`: 3,112 ventanas, 120 características y la etiqueta.
- `Datos modelo/datos_test.csv`: 3,112 ventanas, 120 características y la etiqueta.

## Modelación

El notebook `modelado/modelacion_3_modelos_def.ipynb` compara tres algoritmos compatibles con las características tabulares numéricas:

- Regresión logística, como modelo lineal de referencia.
- SVM con kernel RBF, para representar relaciones no lineales.
- Random Forest, para aprender relaciones no lineales e interacciones entre características.

Los modelos se ajustan con train y se comparan con validation mediante exactitud, precisión macro y F1-score macro. Random Forest obtuvo el mejor desempeño:

| Modelo | Exactitud validation | Precisión macro validation | F1 macro validation |
|---|---:|---:|---:|
| Regresión logística | 0.7751 | 0.7640 | 0.7636 |
| SVM | 0.8602 | 0.8539 | 0.8524 |
| Random Forest | 0.9460 | 0.9453 | 0.9443 |

### Selección de hiperparámetros

La versión definitiva utiliza un **conjunto fijo de validation**, no cross-validation. Cada configuración de Random Forest se ajusta únicamente con train y se evalúa sobre validation mediante F1 macro. La combinación con el mejor resultado se selecciona sin consultar test.

Después de elegir los hiperparámetros, train y validation se unen para ajustar un nuevo Random Forest desde cero. Test se utiliza una sola vez para medir el desempeño final. Esta estrategia es más sencilla que cross-validation y mantiene una separación clara entre ajuste, selección y evaluación, aunque sus resultados pueden depender más de la partición fija de validation.

### Revisión de sobreajuste

Random Forest alcanzó 1.0000 de exactitud y F1 macro en train. En validation obtuvo 0.9460 de exactitud y 0.9443 de F1 macro, con una brecha de F1 de 0.0557. Por ello se identificó una señal moderada de sobreajuste.

Para revisarla se realizaron tres análisis adicionales:

1. Comparación directa entre train y validation.
2. Búsqueda controlada de hiperparámetros usando validation.
3. Curva de profundidad usando train y validation.

Train, validation y test se separan a nivel de señal completa antes de crear ventanas. Esto garantiza que ninguna señal comparta fragmentos entre conjuntos.

La búsqueda seleccionó los mismos parámetros del modelo inicial. La curva mostró subajuste con profundidades pequeñas y una estabilización del F1 macro de validation en **0.9443** desde una profundidad aproximada de 30. Limitar la profundidad no mejoró la generalización.

Después de seleccionar la configuración se unieron train y validation para reajustar el modelo final. En test reservado obtuvo **0.9563 de exactitud**, **0.9549 de precisión macro** y **0.9531 de F1 macro**. Test se utilizó una sola vez al final.

## Reproducibilidad

### 1. Clonar el repositorio

```bash
git clone https://github.com/DiegoVillaa/Concentracion_Reto.git
cd Concentracion_Reto
```

### 2. Crear un entorno virtual

Se recomienda Python 3.11 o una versión posterior.

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter joblib
```

### 4. Ejecutar el ETL

El notebook utiliza rutas relativas, por lo que debe ejecutarse desde la carpeta `ETL`:

```bash
cd ETL
jupyter notebook etl_definitivo.ipynb
```

Dentro de Jupyter se debe seleccionar **Restart Kernel and Run All Cells**. El notebook volverá a generar `datos_train.csv`, `datos_validation.csv` y `datos_test.csv` dentro de `Datos modelo`.

También puede ejecutarse desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace etl_definitivo.ipynb --ExecutePreprocessor.timeout=1200
```

### 5. Ejecutar la modelación

Después de completar el ETL:

```bash
cd ../modelado
jupyter notebook modelacion_3_modelos_def.ipynb
```

Se debe ejecutar el notebook completo y en orden. La búsqueda de hiperparámetros y la curva de validación entrenan varios Random Forest, por lo que esta sección puede tardar algunos minutos.

Desde terminal también puede utilizarse:

```bash
jupyter nbconvert --to notebook --execute --inplace modelacion_3_modelos_def.ipynb --ExecutePreprocessor.timeout=2400
```

## Interfaz web

La carpeta `Interfaz` contiene una aplicación en Streamlit para utilizar el Random Forest definitivo sin ejecutar los notebooks. La aplicación permite:

- Seleccionar una actividad y una repetición de los archivos locales del dataset.
- Cargar manualmente los dos archivos `.npy` correspondientes a los grupos de sensores `_1` y `_2`.
- Unir ambos grupos para obtener una señal de `880 × 12`.
- Dividir la señal en cuatro ventanas consecutivas de `220 × 12`.
- Extraer automáticamente las mismas 120 características utilizadas durante el modelado.
- Mostrar la actividad predicha para cada ventana.
- Visualizar los canales de la señal.
- Consultar los hiperparámetros definitivos y las métricas finales del modelo.

Cada archivo de sensores puede contener una sola repetición con forma `(880, 6)` o varias repeticiones con forma `(n, 880, 6)`. Los archivos `_1` y `_2` deben corresponder a la misma actividad y conservar el mismo orden de repeticiones. La repetición ubicada en la posición `i` de ambos archivos representa el mismo movimiento.

### Instalación de la interfaz

Con el entorno virtual activado y desde la raíz del repositorio:

```bash
python -m pip install -r Interfaz/requirements.txt
```

### Ejecución

```bash
python -m streamlit run Interfaz/app.py
```

Streamlit mostrará una dirección local, normalmente:

```text
http://localhost:8501
```

El archivo `Interfaz/modelo_random_forest.joblib` ya contiene el modelo ajustado con train y validation. Por ello, no es necesario volver a entrenarlo para utilizar la aplicación.

Si se necesita regenerar el archivo del modelo después de volver a ejecutar el ETL, debe utilizarse:

```bash
python Interfaz/entrenar_modelo.py
```

Este script utiliza únicamente `datos_train.csv` y `datos_validation.csv`. El conjunto de test no se incorpora al reentrenamiento.

### Interpretación de los resultados

La interfaz presenta una predicción para cada una de las cuatro ventanas de 220 puntos. La actividad dominante se incluye únicamente como resumen visual. Las métricas reportadas por el proyecto se calcularon tratando cada ventana como una observación independiente, sin aplicar votación para convertirlas en una sola predicción.

## Limitaciones

Los archivos procesados no incluyen identificadores de paciente o sesión por repetición. Por ello, no es posible realizar una separación independiente por sujeto. Los resultados representan la clasificación de nuevas señales o ventanas dentro de la población combinada del conjunto de datos y podrían sobreestimar la generalización a pacientes completamente nuevos.

La comparación de algoritmos y la selección de hiperparámetros se realizan con el conjunto fijo de validation, sin cross-validation. El conjunto de test permanece reservado para una única evaluación final del Random Forest seleccionado.
