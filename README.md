# Documentación del Proyecto: Análisis de Datos Bancarios 
Andres Candelo 

Cientifico de datos
# Descripción del Proyecto

Este proyecto tiene como objetivo realizar un análisis exploratorio de datos (EDA) y luego construir un modelo de clasificación para predecir la variable y en un conjunto de datos bancarios. La variable y indica si un cliente ha aceptado o no una oferta de depósito a plazo fijo.

Las tareas principales son:

Análisis Exploratorio de Datos (EDA): Generar un informe interactivo utilizando Sweetviz para explorar las características del conjunto de datos.

Modelo de Clasificación: Entrenar y evaluar varios modelos de clasificación, incluyendo LGBMClassifier, SVC y RandomForestClassifier, para predecir la variable y.

# Características Principales

Análisis Exploratorio de Datos (EDA): Visualización interactiva de las distribuciones, correlaciones y estadísticas de los datos utilizando Sweetviz.

## Análisis Exploratorio de Datos con Sweetviz

En este proyecto utilizamos la librería **Sweetviz** para realizar un análisis exploratorio de datos (EDA) de nuestro conjunto de datos bancarios. Sweetviz genera un informe visual e interactivo con estadísticas descriptivas, distribuciones de variables, correlaciones y más.

Se crea la siguiente funcion para hacer un llamado a los datos en la url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

```python
import pandas as pd
import zipfile
import os
import requests
from io import BytesIO

def cargar_datos_banco(url):
    # Descargamos el archivo ZIP
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error al descargar el archivo: {response.status_code}")
    
    # Descomprimir el archivo ZIP
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        # Extraemos todos los archivos en un directorio temporal
        zip_ref.extractall("temp_banco")
    
    # Leer los archivos CSV
    bank_full_path = os.path.join("temp_banco", "bank-full.csv")
    bank_path = os.path.join("temp_banco", "bank.csv")
    
    if not os.path.exists(bank_full_path) or not os.path.exists(bank_path):
        raise Exception("No se encontraron los archivos 'bank-full.csv' y/o 'bank.csv' en el archivo ZIP.")
    
    # Cargar los CSV en pandas
    df_bank_full = pd.read_csv(bank_full_path, delimiter=';')
    df_bank = pd.read_csv(bank_path, delimiter=';')
    
    # Limpiar el directorio temporal
    for f in os.listdir("temp_banco"):
        os.remove(os.path.join("temp_banco", f))
    os.rmdir("temp_banco")
    
    # Retornar los dataframes
    return df_bank_full, df_bank


```
A continuación, se muestra el código utilizado para crear y visualizar el reporte:

```python
# Importamos la librería Sweetviz para crear el reporte de análisis exploratorio de datos
import sweetviz as sv

# Creamos un reporte de análisis exploratorio para el dataframe 'df_bank_full'
# Este reporte incluirá estadísticas descriptivas, distribuciones y relaciones de variables
report = sv.analyze(df_bank_full)

# Esto permite ver el análisis visual del conjunto de datos en un formato fácil de interpretar
report.show_notebook()
```
A continuacion se presenta el analisis de algunas variables relevantes para el problema 

![image](https://github.com/user-attachments/assets/4ce680f4-fcc5-4a46-b10e-9029aa3429bd)

Para a variable age podemos observar que la gran mayoria tienen entre 25 y 35 años

![image](https://github.com/user-attachments/assets/24e6a6df-4ad7-4767-880b-036603ceb8e6)

Para la variable job vemos que aproximadamente el 24% de las personas  tienen un job de blue collar, el 22 son management y el 16 % son technician

![image](https://github.com/user-attachments/assets/65e3b41a-78f3-4f05-a729-1cb207c00859)

Podemos observar que para marital el 60% estan casados, el 28% single y el 12% divorced

![image](https://github.com/user-attachments/assets/1c7049d9-7ee7-4872-9bb7-30fe54b9a865)

Para la variable education observamos que el 50% tiene informacion secundaria

Ahora vamos hacer un analisis con respecto a la variable y, para esto vamos a resolver las siguientes preguntas de negocio que nos ayudaran a comprender como se comportan las variables con respecto a y:

Que job tienen los usuarios que mas acceden han aceptado o no una oferta de depósito a plazo fijo?

![image](https://github.com/user-attachments/assets/3ce34fb4-d7ee-4028-a475-98951156f300)

Podemos observar que los que mas han aceptado son los estudiantes, retirados y desempleados. 

Existen meses donde los usuarios son las propensos a aceptar una oferta de depósito a plazo fijo?

![image](https://github.com/user-attachments/assets/53978131-dfbd-44b7-88a7-4794aed7d07e)

Los meses como dec, mar, oct y sep es cuando los clientes han aceptado

Los clientes que tienen un prestamo housing son propensos aceptar una oferta de depósito a plazo fijo?
![image](https://github.com/user-attachments/assets/ec332980-9878-466b-8743-ce155c6c7a1a)

No, observamos que los clientes que no cuentan con housing han acepto mayoritariamente

Una vez se realice el EDA donde se valido que no existen valores atipicos y se comprende como ha funcionado el negocio se va prueban diferentes modelos 

```python
import lazypredict  
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# Aplicar One-Hot Encoding a las columnas categóricas
X = df_bank_full.drop('y', axis=1)  # Excluye la columna objetivo 'y'
y = df_bank_full['y']  # Variable objetivo

# Convertir las variables categóricas a variables dummy
X_encoded = pd.get_dummies(X, drop_first=True)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Inicializar LazyClassifier 
clf = LazyClassifier(verbose=0, ignore_warnings=True) 

# Ajustar los modelos
models, results = clf.fit(X_train, X_test, y_train, y_test)
```
obteniendo el siguiente resultado 
![image](https://github.com/user-attachments/assets/eccab630-08b9-4f32-886b-d58f52f8b387)


Podemos observar que hay 3 modelos muestran un buen performance, estos se van a tomar como base para realizar el preprocesamiento de los datos y la optimización de hiperparametros

Preprocesamiento de Datos: Limpieza de los datos, manejo de valores nulos, codificación de variables categóricas y escalado de características.

```python
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column='y', test_size=0.3, random_state=42):
    # Reemplazar los valores 'yes' por 1 y 'no' por 0 en la columna objetivo
    df[target_column] = df[target_column].replace({'yes': 1, 'no': 0})
    
    # Crear una nueva columna 'age_group' con rangos etarios para la variable 'age'
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], 
                             labels=['0-18', '19-30', '31-45', '46-60', '61+'])
    
    # Eliminar la columna original 'age'
    df = df.drop(columns=['age'])
    
    # Separar las variables predictoras (X) y la variable objetivo (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Seleccionar solo las columnas categóricas
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    
    # Aplicar One Hot Encoding solo a las columnas categóricas
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

```
Se organiza las edades por rangos etareos

Modelos de Clasificación: Entrenamiento de modelos de clasificación utilizando LGBMClassifier y RandomForestClassifier.

```python
from skopt import BayesSearchCV
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def optimize_model(model, param_space, X_train, y_train, n_iter=32, scoring='f1', cv=3, n_jobs=-1, random_state=42):
    """
    Función para optimizar un modelo utilizando optimización bayesiana.

    model: el modelo de machine learning (SVC, LightGBM, RandomForest, etc.)
    param_space: espacio de búsqueda para los hiperparámetros
    X_train: conjunto de entrenamiento
    y_train: etiquetas del conjunto de entrenamiento
    n_iter: número de iteraciones para la optimización
    scoring: métrica de evaluación
    cv: número de pliegues en la validación cruzada
    n_jobs: número de núcleos a usar para la optimización
    random_state: semilla aleatoria

    Returns: modelo optimizado y su mejor puntaje
    """
    # Configurar la búsqueda bayesiana de hiperparámetros
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,             # número de iteraciones para la optimización
        scoring=scoring,          # métrica de evaluación
        cv=cv,                    # número de pliegues en la validación cruzada
        random_state=random_state,
        n_jobs=n_jobs              # utiliza todos los núcleos disponibles
    )

    # Ejecutar la optimización
    opt.fit(X_train, y_train)
    
    # Imprimir los mejores parámetros y el puntaje
    print("Best Parameters:", opt.best_params_)
    print("Best Score:", opt.best_score_)
    
    return opt.best_estimator_
```
Se utiliza una optmizacion bayesiana para realizar el ajuste optimo de los hiperparametros, para esto se crea una funcion que ayude a probar los modelos de random forest y ligthbm.

Evaluación de Modelos: Comparación de modelos mediante métricas como accuracy, precision, recall, F1-score y roc AUC. Los resultados fueron los siguientes:

Para el modelo Random Forest se obtuvo:

F1 Score en el conjunto de prueba: 0.9017988793866116

Accuracy en el conjunto de prueba: 0.9017988793866116

y la curva roc AUC tuvo el siguiente:

![image](https://github.com/user-attachments/assets/7de28e82-9b6a-4515-9a18-23351ab8a9a3)

Se puede observar que el modelo distingue bien entre las clases, es decir que la tasa de verdaderos positivos y la tasa de falsos positivos se encuentra baja. 

Para el modelo LGBMClassifier se obtuvo:

F1 Score en el conjunto de prueba: 0.9017988793866116

Accuracy en el conjunto de prueba: 0.9065909761132409

y la curva roc AUC tuvo la siguiente grafica 

![image](https://github.com/user-attachments/assets/dcfb52ad-3dcb-4aba-881c-68061b2b5e14)


Se puede observar que el modelo distingue bien entre las clases, es decir que la tasa de verdaderos positivos y la tasa de falsos positivos se encuentra baja.

Se concluye que este modelo tiene una mejor distincion entre las clases por lo tanto es el que mejor performance le da al negocio

Para comprender un poco mas sobre la importancia de las variables se calcula el feature importance para el modelo LGTBM, se puede observar en el siguiente grafico:

![image](https://github.com/user-attachments/assets/5025c8da-b213-4e0d-ab73-6dfffbce7051)


duration es la característica más importante por una diferencia considerable. Esto sugiere que el tiempo de duración de la interacción es el factor más determinante para el modelo.

day y pdays también tienen una alta importancia, aunque mucho menor que duration. Esto indica que el momento de contacto (día y tiempo desde el último contacto) es relevante para la efectividad de la campaña.

balance tiene una importancia intermedia, lo que indica que el saldo del cliente puede influir en el éxito de la campaña, pero no tanto como la duración. Un balance positivo podría estar relacionado con una mayor probabilidad de aceptación de productos financieros.

Las variables contact_unknown, month_aug, month_jul, campaign, month_may, y poutcome_success también son importantes, aunque en menor medida. Esto sugiere que factores como el canal de contacto, el mes en el que se realiza el contacto y el éxito de campañas previas también influyen, pero de forma menos crítica en comparación con las primeras variables.






