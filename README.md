# Reto REHAB

Este repositorio contiene el desarrollo de la solución para el reto REHAB de la concentración **Inteligencia Artificial para la Ciencia de Datos**.

## Introducción al reto

Un accidente cerebrovascular puede afectar la movilidad, la fuerza y la coordinación de una persona, por lo que la evaluación y el entrenamiento de rehabilitación son importantes durante su recuperación. El conjunto de datos REHAB contiene registros cinemáticos obtenidos mediante sensores portátiles durante distintos movimientos de rehabilitación.

Para este proyecto se utilizan los datos procesados correspondientes a **16 actividades de entrenamiento**, identificadas con las etiquetas `000` a `015`. El objetivo es preparar las señales y construir un modelo de clasificación capaz de identificar la actividad realizada a partir de las mediciones de los sensores.

## Flujo general del proyecto

El trabajo principal sigue este orden:

1. Exploración inicial de los archivos y las señales.
2. Reconstrucción y validación del archivo procesado `014_1.npy`.
3. Limpieza y preparación de los datos mediante el ETL definitivo.
4. Separación de las señales completas en entrenamiento y prueba.
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
│   └── datos_test.csv
├── modelado/
│   └── modelación_3_modelos.ipynb
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
- Separa las **señales completas** en 80% entrenamiento y 20% prueba mediante estratificación.
- Divide cada señal en cuatro ventanas consecutivas de 220 puntos, sin traslape.
- Calcula diez estadísticas para cada uno de los 12 canales: media, mediana, desviación estándar, mínimo, máximo, rango, percentiles 25 y 75, rango intercuartílico y RMS.
- Genera 120 características por ventana y guarda las tablas finales.

La separación se realiza antes de crear las ventanas. Por lo tanto, las cuatro ventanas procedentes de una señal permanecen juntas en entrenamiento o en prueba, evitando que fragmentos relacionados aparezcan en ambos conjuntos.

Los archivos generados son:

- `Datos modelo/datos_train.csv`: 12,448 ventanas, 120 características y la etiqueta.
- `Datos modelo/datos_test.csv`: 3,112 ventanas, 120 características y la etiqueta.

## Modelación

El notebook `modelado/modelación_3_modelos.ipynb` compara tres algoritmos compatibles con las características tabulares numéricas:

- Regresión logística, como modelo lineal de referencia.
- SVM con kernel RBF, para representar relaciones no lineales.
- Random Forest, para aprender relaciones no lineales e interacciones entre características.

La comparación reporta exactitud, precisión macro y F1-score macro. Random Forest obtuvo el mejor desempeño inicial:

| Modelo | Exactitud | Precisión macro | F1 macro |
|---|---:|---:|---:|
| Regresión logística | 0.7873 | 0.7760 | 0.7750 |
| SVM | 0.8869 | 0.8817 | 0.8797 |
| Random Forest | 0.9569 | 0.9544 | 0.9529 |

### Revisión de sobreajuste

Random Forest alcanzó 1.0000 de exactitud y F1 macro en entrenamiento. En prueba obtuvo 0.9569 de exactitud y 0.9529 de F1 macro, por lo que se identificó una señal moderada de sobreajuste.

Para revisarla se realizaron tres análisis adicionales:

1. Validación cruzada agrupada dentro de entrenamiento.
2. Búsqueda controlada de hiperparámetros.
3. Curva de validación para `max_depth`.

La validación utiliza `StratifiedGroupKFold`. Cada grupo representa las cuatro ventanas de una señal original, de modo que ninguna señal comparte ventanas entre ajuste y validación. Los tres folds obtuvieron cero grupos compartidos.

La búsqueda seleccionó los mismos parámetros del modelo inicial y no mejoró sus métricas. La curva mostró subajuste con profundidades pequeñas y una estabilización del F1 macro de validación en **0.9311** desde una profundidad aproximada de 30. Limitar la profundidad a 10 redujo el F1 de validación a 0.8879, por lo que no mejoró la generalización.

En consecuencia, se conserva la configuración inicial de Random Forest. El desempeño perfecto en entrenamiento se reconoce como una señal que debe vigilarse, pero las estrategias de regularización evaluadas no produjeron una mejora.

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

Dentro de Jupyter se debe seleccionar **Restart Kernel and Run All Cells**. El notebook volverá a generar `datos_train.csv` y `datos_test.csv` dentro de `Datos modelo`.

También puede ejecutarse desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace etl_definitivo.ipynb --ExecutePreprocessor.timeout=1200
```

### 5. Ejecutar la modelación

Después de completar el ETL:

```bash
cd ../modelado
jupyter notebook "modelación_3_modelos.ipynb"
```

Se debe ejecutar el notebook completo y en orden. La búsqueda de hiperparámetros y la curva de validación entrenan varios Random Forest, por lo que esta sección puede tardar algunos minutos.

Desde terminal también puede utilizarse:

```bash
jupyter nbconvert --to notebook --execute --inplace "modelación_3_modelos.ipynb" --ExecutePreprocessor.timeout=1800
```

## Limitaciones

Los archivos procesados no incluyen identificadores de paciente o sesión por repetición. Por ello, no es posible realizar una separación independiente por sujeto. Los resultados representan la clasificación de nuevas señales o ventanas dentro de la población combinada del conjunto de datos y podrían sobreestimar la generalización a pacientes completamente nuevos.

La comparación inicial de los tres algoritmos se reporta sobre el conjunto de prueba disponible. La búsqueda de hiperparámetros y la curva de validación se realizan exclusivamente dentro de entrenamiento mediante grupos de señal.
