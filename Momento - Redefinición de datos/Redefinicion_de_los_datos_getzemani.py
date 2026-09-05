# Redefinición de los datos

import os
import numpy as np
import matplotlib.pyplot as plt


# Ruta donde están los archivos procesados .npy
RUTA_DATOS = "/content/Concentracion_Reto/Rehab_exercise/d02_processed_data"

# Frecuencia de muestreo de los sensores
FS = 50.0

# Longitud mínima que debe tener una repetición para usarla
MIN_LEN = 100


# ---------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------

# Aquí se guardarán todos los archivos que sí puedan abrirse
datos = {}

# Hay 16 movimientos y 2 archivos por movimiento
for movimiento in range(16):
    for sensor in (1, 2):

        archivo = os.path.join(
            RUTA_DATOS,
            f"{movimiento:03d}_{sensor}.npy"
        )

        try:
            datos[(movimiento, sensor)] = np.load(archivo)

        # Si un archivo está dañado o no existe, simplemente no se agrega
        except:
            pass


# ---------------------------------------------------------
# 2. LIMPIEZA DE LAS SEÑALES
# ---------------------------------------------------------

def longitud_real(x):
    # Calcula cuánto cambia la señal entre una muestra y otra
    d = np.abs(np.diff(x, axis=0)).max(axis=1)

    # Busca hasta dónde hay movimiento real
    pos = np.where(d > 1e-9)[0]

    if len(pos) == 0:
        return 0

    return int(pos[-1] + 2)


def media_movil(x, k):
    # Si la señal es muy corta, solo se centra
    if len(x) < k:
        return x - x.mean()

    # Calcula una media móvil usando suma acumulada
    c = np.cumsum(np.insert(x, 0, 0.0))

    return (c[k:] - c[:-k]) / k


def preparar(x):
    # Detecta la longitud que realmente contiene información
    L = longitud_real(x)

    # Si la señal es demasiado corta, no se utiliza
    if L < MIN_LEN:
        return None

    # Se elimina el relleno que queda al final
    x = x[:L].astype(float)

    # Se centra cada canal alrededor de cero
    x = x - x.mean(axis=0, keepdims=True)

    return x


# ---------------------------------------------------------
# 3. EXTRACCIÓN DE ATRIBUTOS
# ---------------------------------------------------------

def atributos_canal(x):
    # Cantidad de muestras
    n = len(x)

    # Desviación estándar
    sd = float(x.std())

    # Desviación absoluta media
    mad = float(np.abs(x).mean())

    # Percentiles para medir dispersión y rango
    q25, q75 = np.percentile(x, [25, 75])
    p05, p95 = np.percentile(x, [5, 95])

    # Si la señal tiene variación, calcula asimetría y curtosis
    if sd > 1e-12:
        z = x / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3)

    else:
        skew = 0.0
        kurt = 0.0

    # Primera diferencia = cambio aproximado entre muestras
    dx = np.diff(x)

    # Segunda diferencia = cambio de la velocidad
    ddx = np.diff(dx)

    vel_mav = float(np.abs(dx).mean() * FS) if len(dx) else 0.0
    vel_std = float(dx.std() * FS) if len(dx) else 0.0
    acc_mav = float(np.abs(ddx).mean() * FS ** 2) if len(ddx) else 0.0

    # Cambios de signo para estimar ciclos
    s = np.sign(x)
    s[s == 0] = 1

    cruces = int((np.abs(np.diff(s)) > 0).sum())

    # Frecuencia aproximada del movimiento
    ciclos = (cruces / 2) / (n / FS)

    # Autocorrelación simple entre muestras consecutivas
    if sd > 1e-12 and n > 2:
        ac1 = float((x[:-1] * x[1:]).mean() / sd ** 2)
    else:
        ac1 = 0.0

    # Señal suavizada para medir qué tan estable es
    sm = media_movil(x, max(2, int(0.5 * FS)))

    if sd > 1e-12:
        smooth = float(sm.var() / sd ** 2)
    else:
        smooth = 0.0

    # Se regresan 12 atributos por canal
    return [
        sd,
        mad,
        float(q75 - q25),
        float(p95 - p05),
        skew,
        kurt,
        vel_mav,
        vel_std,
        acc_mav,
        ciclos,
        ac1,
        smooth
    ]


def correlacion(a, b):
    # Mide qué tan relacionadas están dos señales
    sa = a.std()
    sb = b.std()

    if sa < 1e-12 or sb < 1e-12:
        return 0.0

    return float(
        ((a - a.mean()) * (b - b.mean())).mean()
        / (sa * sb)
    )


# ---------------------------------------------------------
# 4. CONSTRUCCIÓN DEL DATASET
# ---------------------------------------------------------

X_lista = []
y_lista = []
indices = []

for movimiento in range(16):

    # Si falta alguno de los sensores del movimiento, se omite
    if (movimiento, 1) not in datos or (movimiento, 2) not in datos:
        continue

    imu = datos[(movimiento, 1)]
    glove = datos[(movimiento, 2)]

    # Se usa la cantidad de repeticiones disponible en ambos archivos
    for i in range(min(len(imu), len(glove))):

        # Limpia cada repetición
        a = preparar(imu[i])
        b = preparar(glove[i])

        if a is None or b is None:
            continue

        fila = []

        # 12 atributos de cada uno de los 6 canales IMU
        for c in range(6):
            fila += atributos_canal(a[:, c])

        # 12 atributos de cada uno de los 6 canales del guante
        for c in range(6):
            fila += atributos_canal(b[:, c])

        # Correlaciones entre los primeros 5 canales del guante
        pares = []

        for p in range(5):
            for q in range(p + 1, 5):
                pares.append(
                    correlacion(b[:, p], b[:, q])
                )

        # Energía aproximada de cada grupo de sensores
        energia_g = float(b[:, :5].std(axis=0).mean())
        energia_i = float(a.std(axis=0).mean())

        # Se agregan 6 relaciones extras
        fila += [
            correlacion(a[:, 0], a[:, 3]),
            correlacion(a[:, 1], a[:, 4]),
            correlacion(a[:, 2], a[:, 5]),
            float(np.mean(pares)),
            correlacion(b[:, :5].mean(axis=1), b[:, 5]),
            float(
                np.log10(
                    (energia_g + 1e-6)
                    / (energia_i + 1e-6)
                )
            )
        ]

        X_lista.append(fila)
        y_lista.append(movimiento)
        indices.append(i)


# Se convierten las listas a arreglos de NumPy
X = np.array(X_lista, dtype=float)
y_original = np.array(y_lista)
indices = np.array(indices)


# ---------------------------------------------------------
# 5. PREPARACIÓN DE LAS CLASES
# ---------------------------------------------------------

# Lista de movimientos disponibles
clases = np.array(sorted(set(y_original)))

# Convierte las clases originales a números consecutivos
mapa = {
    clase: i
    for i, clase in enumerate(clases)
}

y = np.array([
    mapa[v]
    for v in y_original
])


# ---------------------------------------------------------
# 6. DIVISIÓN TRAIN / VALIDATION / TEST
# ---------------------------------------------------------

split = np.empty(len(y), dtype=object)

for clase in np.unique(y):

    # Busca todas las filas de esa clase
    pos = np.where(y == clase)[0]

    # Mantiene el orden original de las repeticiones
    pos = pos[np.argsort(indices[pos])]

    n = len(pos)

    # 60% entrenamiento
    a = int(n * 0.60)

    # 20% validación
    b = int(n * 0.80)

    split[pos[:a]] = "train"
    split[pos[a:b]] = "val"
    split[pos[b:]] = "test"


# Guarda los atributos por si se quieren revisar después
np.savetxt(
    "rehab_features.csv",
    np.column_stack((y_original, X)),
    delimiter=","
)


# ---------------------------------------------------------
# 7. FUNCIONES PARA SOFTMAX
# ---------------------------------------------------------

def ajustar_escalador(X):
    # Calcula media y desviación estándar del entrenamiento
    mu = X.mean(axis=0)
    sd = X.std(axis=0)

    # Evita divisiones entre cero
    sd[sd < 1e-12] = 1

    return mu, sd


def escalar(X, mu, sd):
    # Estandariza cada atributo
    return (X - mu) / sd


def con_sesgo(X):
    # Agrega una columna de 1 para el término independiente
    return np.hstack([
        np.ones((len(X), 1)),
        X
    ])


def one_hot(y, k):
    # Convierte cada clase a un vector como [0, 1, 0, ...]
    Y = np.zeros((len(y), k))

    Y[np.arange(len(y)), y] = 1

    return Y


def softmax(z):
    # Evita valores demasiado grandes antes de usar exp()
    z = z - z.max(axis=1, keepdims=True)

    e = np.exp(z)

    # Cada fila termina sumando 1
    return e / e.sum(axis=1, keepdims=True)


def accuracy(y_real, y_pred):
    # Porcentaje de predicciones correctas
    return float(
        (y_real == y_pred).mean()
    )


def matriz_confusion(y_real, y_pred, k):
    # Matriz donde filas = clase real y columnas = predicción
    m = np.zeros((k, k), dtype=int)

    for real, pred in zip(y_real, y_pred):
        m[real, pred] += 1

    return m


def f1_macro(y_real, y_pred, k):
    # Calcula F1 por clase y después obtiene el promedio
    valores = []

    for clase in range(k):

        tp = np.sum(
            (y_real == clase)
            & (y_pred == clase)
        )

        fp = np.sum(
            (y_real != clase)
            & (y_pred == clase)
        )

        fn = np.sum(
            (y_real == clase)
            & (y_pred != clase)
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        valores.append(f1)

    return float(np.mean(valores))


# ---------------------------------------------------------
# 8. ENTRENAMIENTO DEL MODELO SOFTMAX
# ---------------------------------------------------------

def entrenar(
    Xtr,
    ytr,
    Xva,
    yva,
    k,
    lam=0.03,
    max_epocas=400
):

    # Semilla para obtener resultados repetibles
    rng = np.random.default_rng(7)

    Xtr = con_sesgo(Xtr)
    Xva = con_sesgo(Xva)

    # Pesos iniciales
    W = np.zeros((Xtr.shape[1], k))

    # Velocidad usada por momentum
    V = np.zeros_like(W)

    # Clases de entrenamiento en formato one-hot
    Ytr = one_hot(ytr, k)

    # Learning rate inicial
    lr = 0.5

    mejor_acc = 0
    mejor_W = W.copy()
    mejor_epoca = 0

    # Se guardan valores para graficar después
    historial_train = []
    historial_val = []

    for epoca in range(1, max_epocas + 1):

        # Mezcla el orden de entrenamiento en cada época
        orden = rng.permutation(len(Xtr))

        # Mini-batches de 128 ejemplos
        for inicio in range(0, len(Xtr), 128):

            b = orden[inicio:inicio + 128]

            # Momentum
            W2 = W + 0.9 * V

            # Probabilidades del modelo
            P = softmax(Xtr[b] @ W2)

            # Gradiente
            grad = Xtr[b].T @ (P - Ytr[b]) / len(b)

            # Regularización L2 excepto en el sesgo
            grad[1:] += lam * W2[1:]

            # Actualización
            V = 0.9 * V - lr * grad
            W = W + V

        # Reduce poco a poco el learning rate
        lr *= 0.995

        # Accuracy de entrenamiento
        pred_train = softmax(Xtr @ W).argmax(axis=1)
        acc_train = accuracy(ytr, pred_train)

        # Accuracy de validación
        pred_val = softmax(Xva @ W).argmax(axis=1)
        acc_val = accuracy(yva, pred_val)

        historial_train.append(acc_train)
        historial_val.append(acc_val)

        # Guarda el mejor modelo encontrado
        if acc_val > mejor_acc:
            mejor_acc = acc_val
            mejor_W = W.copy()
            mejor_epoca = epoca

        # Early stopping
        if epoca - mejor_epoca >= 60:
            break

    return (
        mejor_W,
        historial_train,
        historial_val
    )


# Máscaras de cada división
train = split == "train"
val = split == "val"
test = split == "test"


# Se ajusta el escalado usando solamente entrenamiento
mu, sd = ajustar_escalador(X[train])

X_train = escalar(X[train], mu, sd)
X_val = escalar(X[val], mu, sd)
X_test = escalar(X[test], mu, sd)


# Entrenamiento principal
W, hist_train, hist_val = entrenar(
    X_train,
    y[train],
    X_val,
    y[val],
    len(clases)
)


def predecir_prob(X_datos):
    # Calcula las probabilidades de todas las clases
    return softmax(
        con_sesgo(X_datos) @ W
    )


# Predicciones
pred_train = predecir_prob(X_train).argmax(axis=1)
pred_val = predecir_prob(X_val).argmax(axis=1)
pred_test = predecir_prob(X_test).argmax(axis=1)


# Accuracy
acc_train = accuracy(y[train], pred_train)
acc_val = accuracy(y[val], pred_val)
acc_test = accuracy(y[test], pred_test)

# F1 del conjunto de prueba
f1_test = f1_macro(
    y[test],
    pred_test,
    len(clases)
)


print("Filas:", len(X))
print("Atributos:", X.shape[1])

print(
    "Accuracy train:",
    round(acc_train, 4)
)

print(
    "Accuracy val:",
    round(acc_val, 4)
)

print(
    "Accuracy test:",
    round(acc_test, 4)
)

print(
    "F1 test:",
    round(f1_test, 4)
)


# ---------------------------------------------------------
# 9. GRÁFICA DEL ENTRENAMIENTO
# ---------------------------------------------------------

plt.figure()

plt.plot(
    hist_train,
    label="Train"
)

plt.plot(
    hist_val,
    label="Validation"
)

plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Entrenamiento Softmax")
plt.legend()

plt.show()


# ---------------------------------------------------------
# 10. MATRIZ DE CONFUSIÓN
# ---------------------------------------------------------

mc = matriz_confusion(
    y[test],
    pred_test,
    len(clases)
)

plt.figure()

plt.imshow(mc)

plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Clase real")

plt.colorbar()
plt.show()


# ---------------------------------------------------------
# 11. VALIDACIÓN CRUZADA
# ---------------------------------------------------------

def crear_folds(y, n_folds=5, mezclar=False):
    # Esta función divide cada clase en varios grupos
    rng = np.random.default_rng(7)

    folds = [
        []
        for _ in range(n_folds)
    ]

    for clase in np.unique(y):

        pos = np.where(y == clase)[0]

        # Permite comparar una división ordenada y una mezclada
        if mezclar:
            pos = rng.permutation(pos)

        partes = np.array_split(
            pos,
            n_folds
        )

        for i, parte in enumerate(partes):
            folds[i].extend(
                parte.tolist()
            )

    return [
        np.array(fold)
        for fold in folds
    ]


def validacion_cruzada(
    X,
    y,
    n_folds=5,
    mezclar=False
):
    # Crea los folds
    folds = crear_folds(
        y,
        n_folds=n_folds,
        mezclar=mezclar
    )

    resultados = []

    for i in range(n_folds):

        # Fold actual como prueba
        test_idx = folds[i]

        # Fold siguiente como validación
        val_idx = folds[
            (i + 1) % n_folds
        ]

        # Los demás se usan para entrenamiento
        train_idx = np.concatenate([
            folds[j]
            for j in range(n_folds)
            if j != i
            and j != (i + 1) % n_folds
        ])

        # Escalado del fold
        mu_cv, sd_cv = ajustar_escalador(
            X[train_idx]
        )

        Xtr = escalar(
            X[train_idx],
            mu_cv,
            sd_cv
        )

        Xva = escalar(
            X[val_idx],
            mu_cv,
            sd_cv
        )

        Xte = escalar(
            X[test_idx],
            mu_cv,
            sd_cv
        )

        # Entrena el modelo del fold
        W_cv, _, _ = entrenar(
            Xtr,
            y[train_idx],
            Xva,
            y[val_idx],
            len(clases),
            max_epocas=250
        )

        # Predice el fold de prueba
        pred = softmax(
            con_sesgo(Xte) @ W_cv
        ).argmax(axis=1)

        acc = accuracy(
            y[test_idx],
            pred
        )

        resultados.append(acc)

    return np.array(resultados)


# Validación cruzada conservando el orden
cv_contigua = validacion_cruzada(
    X,
    y,
    n_folds=5,
    mezclar=False
)

# Validación cruzada mezclando las filas
cv_mezclada = validacion_cruzada(
    X,
    y,
    n_folds=5,
    mezclar=True
)


print(
    "CV contigua:",
    np.round(cv_contigua, 4)
)

print(
    "Promedio CV contigua:",
    round(float(cv_contigua.mean()), 4)
)

print(
    "CV mezclada:",
    np.round(cv_mezclada, 4)
)

print(
    "Promedio CV mezclada:",
    round(float(cv_mezclada.mean()), 4)
)


# Gráfica de los folds
x_folds = np.arange(
    1,
    len(cv_contigua) + 1
)

plt.figure()

plt.plot(
    x_folds,
    cv_contigua,
    marker="o",
    label="Contigua"
)

plt.plot(
    x_folds,
    cv_mezclada,
    marker="o",
    label="Mezclada"
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Validación cruzada")
plt.legend()

plt.show()


# ---------------------------------------------------------
# 12. k-NN DESDE CERO
# ---------------------------------------------------------

def distancia_euclidiana(a, b):
    # Distancia entre dos filas de atributos
    return np.sqrt(
        ((a - b) ** 2).sum()
    )


def knn_predecir(
    X_train,
    y_train,
    X_test,
    k=5
):
    predicciones = []

    for fila in X_test:

        # Calcula distancia contra todas las filas de entrenamiento
        distancias = np.sqrt(
            ((X_train - fila) ** 2).sum(axis=1)
        )

        # Busca los k vecinos más cercanos
        vecinos = np.argsort(
            distancias
        )[:k]

        # Obtiene sus clases
        clases_vecinas = y_train[
            vecinos
        ]

        # Cuenta cuántas veces aparece cada clase
        conteos = np.bincount(
            clases_vecinas,
            minlength=len(clases)
        )

        # La clase más repetida es la predicción
        predicciones.append(
            conteos.argmax()
        )

    return np.array(
        predicciones
    )


# Se usa el mismo escalado que Softmax
pred_knn = knn_predecir(
    X_train,
    y[train],
    X_test,
    k=5
)

acc_knn = accuracy(
    y[test],
    pred_knn
)

f1_knn = f1_macro(
    y[test],
    pred_knn,
    len(clases)
)


print(
    "Accuracy Softmax:",
    round(acc_test, 4)
)

print(
    "Accuracy k-NN:",
    round(acc_knn, 4)
)

print(
    "F1 Softmax:",
    round(f1_test, 4)
)

print(
    "F1 k-NN:",
    round(f1_knn, 4)
)


# ---------------------------------------------------------
# 13. GRÁFICA SOFTMAX VS k-NN
# ---------------------------------------------------------

modelos = [
    "Softmax",
    "k-NN"
]

accuracies = [
    acc_test,
    acc_knn
]

plt.figure()

plt.bar(
    modelos,
    accuracies
)

plt.ylabel("Accuracy")
plt.title("Comparación de modelos")

plt.ylim(0, 1)

plt.show()


# ---------------------------------------------------------
# 14. PREDICCIÓN INDIVIDUAL
# ---------------------------------------------------------

def predecir(indice):
    # Obtiene probabilidades de una fila de prueba
    p = predecir_prob(
        X_test[indice:indice + 1]
    )[0]

    # Clase con mayor probabilidad
    pred = clases[
        p.argmax()
    ]

    # Clase real
    real = clases[
        y[test][indice]
    ]

    print("Real:", real)
    print("Predicción:", pred)

    print(
        "Confianza:",
        round(
            float(p.max()) * 100,
            2
        ),
        "%"
    )

    return pred


# Ejemplo de predicción
predecir(0)
