"""
RECONSTRUCCIÓN Y VALIDACIÓN DEL ARCHIVO PROCESADO 014_1.npy
=======================================================================

1. Contexto del problema
------------------------
El archivo ``014_1.npy`` incluido en ``d02_processed_data`` no puede
cargarse como un arreglo NumPy porque su contenido binario está dañado.
El problema se mantuvo después de descargar nuevamente el archivo desde
la fuente oficial, por lo que no corresponde a un error de la ruta ni de
la función ``np.load``.

El archivo ``014_1.npy`` de ``d01_raw_data`` sí puede abrirse y contiene
359 muestras, 880 puntos temporales y 6 canales. Por eso se decidió
reconstruir su versión procesada a partir de los datos raw.


2. Objetivo del programa
------------------------
El objetivo no es inventar valores ni aproximar manualmente la actividad
014. El programa busca reproducir el procedimiento utilizado en los
archivos processed oficiales y comprobarlo primero con actividades para
las que existen tanto el archivo raw como el processed válido.

Solamente si la transformación coincide numéricamente con todos los
archivos oficiales de referencia se permite reconstruir ``014_1.npy``.


3. Archivos utilizados para validar la transformación
------------------------------------------------------
Se utilizan seis archivos oficiales válidos:

    - 000_1.npy y 000_2.npy
    - 001_1.npy y 001_2.npy
    - 013_1.npy y 013_2.npy

Para cada uno se realiza el siguiente procedimiento:

    a. Se carga el archivo de ``d01_raw_data``.
    b. Se aplica la transformación identificada.
    c. Se carga el archivo oficial de ``d02_processed_data``.
    d. Se compara el archivo reconstruido con el processed oficial.
    e. Se calculan la diferencia máxima y el error absoluto medio.

Los archivos reconstruidos de validación se mantienen en memoria. Este
programa no sobrescribe los archivos raw ni los processed oficiales.


4. Transformación identificada
------------------------------
La transformación se aplica por separado a cada muestra y canal:

    1. Filtro de media móvil con una ventana de 10 puntos.
    2. Uso de ``np.convolve(..., mode="same")`` para conservar los 880
       puntos temporales y aplicar relleno con ceros en los bordes.
    3. Cálculo de la media de los 880 puntos de cada canal dentro de cada
       muestra.
    4. Resta de esa media para obtener una señal con media cercana a cero.

La forma del arreglo no cambia durante el procesamiento:

    (número de muestras, 880 puntos, 6 canales)


5. Igualdad exacta e igualdad numérica
--------------------------------------
Se calculan dos comparaciones diferentes:

``np.array_equal``
    Exige que todos los valores sean idénticos hasta el último bit. Puede
    devolver ``False`` por diferencias microscópicas de redondeo en
    operaciones con números ``float64``.

``np.allclose``
    Comprueba que las diferencias estén dentro de una tolerancia numérica.
    En este programa se utilizan ``rtol=1e-10`` y ``atol=1e-10``.

Durante las pruebas, los seis archivos presentaron igualdad numérica. Las
diferencias máximas fueron del orden de 10^-13 y los errores medios del
orden de 10^-14. Estas diferencias corresponden a redondeo de punto
flotante y no representan diferencias relevantes en las señales.

El programa utiliza la igualdad numérica como condición para continuar.
Si uno de los seis archivos no cumple esta condición, se genera un error
y ``014_1.npy`` no se reconstruye.


6. Reconstrucción de 014_1.npy
------------------------------
Después de validar la transformación:

    a. Se carga ``d01_raw_data/014_1.npy``.
    b. Se carga ``d02_processed_data/014_2.npy``.
    c. Se comprueba que ambos tengan la misma forma y, por lo tanto, la
       misma cantidad de muestras.
    d. Se aplica a raw ``014_1.npy`` la transformación ya validada.
    e. Se guarda el resultado como un archivo nuevo.
    f. El archivo guardado se vuelve a cargar con ``allow_pickle=False``
       para verificar que sea un archivo NumPy válido.

El archivo corrupto original no se sobrescribe.


7. Estructura de carpetas esperada
----------------------------------
Los datos raw y processed se encuentran en la carpeta central
``Rehab_exercise``, que también es utilizada por el EDA y el ETL. De esta
manera se evita conservar una segunda copia de los datos dentro de la
carpeta de reconstrucción:

    Concentracion_Reto/
    ├── Rehab_exercise/
    │   ├── d01_raw_data/
    │   │   ├── 000_1.npy
    │   │   ├── 000_2.npy
    │   │   ├── 001_1.npy
    │   │   ├── 001_2.npy
    │   │   ├── 013_1.npy
    │   │   ├── 013_2.npy
    │   │   └── 014_1.npy
    │   └── d02_processed_data/
    │       ├── 000_1.npy
    │       ├── 000_2.npy
    │       ├── 001_1.npy
    │       ├── 001_2.npy
    │       ├── 013_1.npy
    │       ├── 013_2.npy
    │       └── 014_2.npy
    └── Reconstrucción de archivo corrupto/
        ├── reconstruir_014_1.py
        └── resultados_reconstruccion/

El archivo ``REHAB.rar`` y la carpeta temporal ``extraido`` no son
necesarios para ejecutar este programa una vez que los archivos raw de
referencia se encuentran en la carpeta central ``Rehab_exercise``.


8. Archivos generados
---------------------
El programa crea la carpeta ``resultados_reconstruccion`` y guarda:

``014_1.npy``
    Versión procesada reconstruida a partir del archivo raw válido.

``validacion_transformacion.csv``
    Evidencia de la comparación de los seis archivos de referencia. El
    CSV incluye formas, igualdad exacta, igualdad numérica, diferencia
    máxima y error absoluto medio.


9. Ejecución
------------
Desde una terminal se puede ejecutar:

    python reconstruir_014_1.py

Antes de incorporar el resultado al dataset del proyecto se recomienda
conservar el archivo corrupto original como respaldo y documentar que la
versión utilizada fue reconstruida y validada numéricamente.
"""

from pathlib import Path
import csv
import numpy as np

BASE = Path(__file__).resolve().parent
DATOS = BASE.parent / "Rehab_exercise"
RAW = DATOS / "d01_raw_data"
PROCESSED = DATOS / "d02_processed_data"
SALIDA = BASE / "resultados_reconstruccion"
SALIDA.mkdir(exist_ok=True)


def transformar_raw(datos_raw):
    """Aplica media móvil de 10 puntos y resta la media por muestra/canal."""
    kernel = np.ones(10, dtype=np.float64) / 10
    datos_filtrados = np.empty_like(datos_raw, dtype=np.float64)

    # Procesamos por separado cada muestra y cada canal.
    for muestra in range(datos_raw.shape[0]):
        for canal in range(datos_raw.shape[2]):
            datos_filtrados[muestra, :, canal] = np.convolve(
                datos_raw[muestra, :, canal], kernel, mode="same"
            )

    # Normalización a media cero sobre los 880 puntos temporales.
    medias = np.mean(datos_filtrados, axis=1, keepdims=True)
    return datos_filtrados - medias


# Comprobamos la transformación con varios archivos oficiales válidos.
archivos_validacion = [
    "000_1.npy", "000_2.npy",
    "001_1.npy", "001_2.npy",
    "013_1.npy", "013_2.npy",
]

resultados = []
print("VALIDACIÓN DE LA TRANSFORMACIÓN\n")

for nombre in archivos_validacion:
    raw = np.load(RAW / nombre, allow_pickle=False)
    processed_oficial = np.load(PROCESSED / nombre, allow_pickle=False)
    reconstruido = transformar_raw(raw)
    diferencia = np.abs(reconstruido - processed_oficial)

    resultado = {
        "archivo": nombre,
        "shape_raw": str(raw.shape),
        "shape_processed": str(processed_oficial.shape),
        "igualdad_exacta": np.array_equal(reconstruido, processed_oficial),
        "igualdad_numerica": np.allclose(
            reconstruido, processed_oficial, rtol=1e-10, atol=1e-10
        ),
        "diferencia_maxima": float(np.max(diferencia)),
        "error_absoluto_medio": float(np.mean(diferencia)),
    }
    resultados.append(resultado)

    print(f"Archivo: {nombre}")
    print(f"  Igualdad exacta: {resultado['igualdad_exacta']}")
    print(f"  Igualdad numérica: {resultado['igualdad_numerica']}")
    print(f"  Diferencia máxima: {resultado['diferencia_maxima']:.3e}")
    print(f"  Error absoluto medio: {resultado['error_absoluto_medio']:.3e}\n")

if not all(resultado["igualdad_numerica"] for resultado in resultados):
    raise RuntimeError(
        "La transformación no coincide con todos los archivos oficiales; "
        "014_1.npy no será reconstruido."
    )


# Reconstruimos 014_1 solo después de superar las validaciones.
raw_014_1 = np.load(RAW / "014_1.npy", allow_pickle=False)
processed_014_2 = np.load(PROCESSED / "014_2.npy", allow_pickle=False)

if raw_014_1.shape != processed_014_2.shape:
    raise ValueError(
        "raw 014_1 y processed 014_2 no tienen la misma forma: "
        f"{raw_014_1.shape} frente a {processed_014_2.shape}."
    )

reconstruido_014_1 = transformar_raw(raw_014_1)
ruta_reconstruido = SALIDA / "014_1.npy"
np.save(ruta_reconstruido, reconstruido_014_1)

# Verificamos que el archivo nuevo pueda abrirse sin pickle y sin cambios.
comprobacion = np.load(ruta_reconstruido, allow_pickle=False)
if not np.array_equal(reconstruido_014_1, comprobacion):
    raise RuntimeError("El archivo guardado no coincide con el arreglo generado.")


# Guardamos evidencia numérica de las comparaciones realizadas.
ruta_reporte = SALIDA / "validacion_transformacion.csv"
with ruta_reporte.open("w", newline="", encoding="utf-8") as archivo_csv:
    escritor = csv.DictWriter(archivo_csv, fieldnames=resultados[0].keys())
    escritor.writeheader()
    escritor.writerows(resultados)

print("RECONSTRUCCIÓN COMPLETADA")
print("Shape de raw 014_1:", raw_014_1.shape)
print("Shape de processed 014_2:", processed_014_2.shape)
print("Shape del 014_1 reconstruido:", reconstruido_014_1.shape)
print(
    "Mayor media absoluta por muestra/canal:",
    np.max(np.abs(reconstruido_014_1.mean(axis=1)))
)
print("Archivo reconstruido:", ruta_reconstruido)
print("Reporte de validación:", ruta_reporte)
