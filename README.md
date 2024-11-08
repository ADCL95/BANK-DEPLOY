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

Preprocesamiento de Datos: Limpieza de los datos, manejo de valores nulos, codificación de variables categóricas y escalado de características.

Modelos de Clasificación: Entrenamiento de modelos de clasificación utilizando LGBMClassifier, SVC y RandomForestClassifier.

Evaluación de Modelos: Comparación de modelos mediante métricas como accuracy, precision, recall, F1-score y roc AUC.
```
obteniendo el siguiente resultado 
![image](https://github.com/user-attachments/assets/eccab630-08b9-4f32-886b-d58f52f8b387)

Podemos observar que hay 3 modelos muestran un buen performance, estos se van a tomar como base para realizar el preprocesamiento de los datos y la optimización de hiperparametros




