# Reto REHAB

Este repositorio contiene el desarrollo de la solución para el reto REHAB de la concentración **Inteligencia Artificial para la Ciencia de Datos**.

## Introducción al reto

Un accidente cerebrovascular puede afectar la movilidad, la fuerza y la coordinación de una persona, por lo que la evaluación y el entrenamiento de rehabilitación son importantes durante su recuperación. El conjunto de datos REHAB contiene registros cinemáticos obtenidos mediante sensores portátiles durante distintos movimientos de rehabilitación.

Para este reto se utilizan los datos procesados correspondientes a 16 actividades de entrenamiento. El objetivo es explorar y preparar estas señales para desarrollar un modelo de clasificación capaz de identificar la actividad realizada a partir de las mediciones de los sensores.

## ETL

Después de revisar y discutir las propuestas individuales, como equipo definimos el ETL que utilizaremos para preparar nuestro conjunto de datos. El procedimiento acordado se encuentra en la carpeta `ETL`, dentro del archivo `etl_definitivo.ipynb`.

## Modelación

En la carpeta `modelado` se incluye el notebook con la implementación de tres modelos de clasificación, junto con su evaluación y la selección del modelo con mejor desempeño.
