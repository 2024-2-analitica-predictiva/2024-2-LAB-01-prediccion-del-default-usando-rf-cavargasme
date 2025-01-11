# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd

raw_train_df = pd.read_csv(
    "../files/input/train_data.csv.zip",
    compression="zip",
)
raw_test_df = pd.read_csv(
    "../files/input/test_data.csv.zip",
    compression="zip",
)

def preprocess_data(df):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.loc[(df["EDUCATION"] != 0)]
    df = df.loc[(df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df

cleaned_train_df = preprocess_data(raw_train_df)
cleaned_test_df = preprocess_data(raw_test_df)
    
import os
import pickle

x_train = cleaned_train_df.drop(columns=["default"])
y_train = cleaned_train_df["default"]

x_test = cleaned_test_df.drop(columns=["default"])
y_test = cleaned_test_df["default"]


if not os.path.exists("./files/grading/"):
    os.makedirs("./files/grading/")
    

with open("./files/grading/x_train.pkl", "wb") as f:
    pickle.dump(x_train, f)

with open("./files/grading/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("./files/grading/x_test.pkl", "wb") as f:
    pickle.dump(x_test, f)

with open("./files/grading/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif

# Columnas categóricas

#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).

pipeline = Pipeline(
    
    [
        (
            "transformer",
            ColumnTransformer(
                [
                    (
                        "encoder",
                        OneHotEncoder(),
                        ["SEX", "EDUCATION", "MARRIAGE"]
                    ),
                ],
                remainder="passthrough"
            ),
        ),
        ("model", RandomForestClassifier()),
    ]
)


import warnings
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

param_grid = {
    "model__n_estimators": [200],
    "model__max_features": ["sqrt"],
    "model__max_depth": [None],
    "model__min_samples_split": [11],
    "model__min_samples_leaf": [2],
}

grid_search_pipeline = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
)

grid_search_pipeline.fit(x_train, y_train)

print(grid_search_pipeline.best_estimator_)
print(grid_search_pipeline.score(x_train, y_train))
print(grid_search_pipeline.score(x_test, y_test))

import os
import pickle
import gzip

if not os.path.exists("../files/models"):
    os.makedirs("../files/models")
    
with open("../files/models/model.pkl.gz", "wb") as file:
    pickle.dump(grid_search_pipeline, file)



import os

from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

if not os.path.exists("../files/output"):
    os.makedirs("../files/output")


metrics = {
    "type":"metrics",
    "dataset":"train",
    "precision": float(precision_score(y_train, grid_search_pipeline.predict(x_train))),
    "balanced_accuracy": float(balanced_accuracy_score(y_train, grid_search_pipeline.predict(x_train))),
    "recall": float(recall_score(y_train, grid_search_pipeline.predict(x_train))),
    "f1-score": float(f1_score(y_train, grid_search_pipeline.predict(x_train))),
}


with open("../files/output/metrics.json", "w") as file:
    file.write(str(metrics).replace("'", '"'))
    file.write("\n")

display(metrics)

metrics = {
    "type":"metrics",
    "dataset":"test",
    "precision": float(precision_score(y_test, grid_search_pipeline.predict(x_test))),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, grid_search_pipeline.predict(x_test))),
    "recall": float(recall_score(y_test, grid_search_pipeline.predict(x_test))),
    "f1-score": float(f1_score(y_test, grid_search_pipeline.predict(x_test))),
}


with open("../files/output/metrics.json", "w") as file:
    file.write(str(metrics).replace("'", '"'))
    file.write("\n")

display(metrics)

from sklearn.metrics import confusion_matrix

cm_train = pd.DataFrame(
    data = confusion_matrix(y_train, grid_search_pipeline.predict(x_train)),
    index = ["True 0", "True 1"],
    columns = ["Predicted 0", "Predicted 1"],
)

metrics = {
    "type":"cm_matrix",
    "dataset":"train",
    "true_0": {
        "predicted_0": int(cm_train.loc["True 0", "Predicted 0"]),
        "predicted_1": int(cm_train.loc["True 0", "Predicted 1"]),
    },
    "true_1": {
        "predicted_0": int(cm_train.loc["True 1", "Predicted 0"]),
        "predicted_1": int(cm_train.loc["True 1", "Predicted 1"]),        
    },
}


with open("../files/output/metrics.json", "a") as file:
    file.write(str(metrics).replace("'", '"'))
    file.write("\n")

display(metrics)




cm_test = pd.DataFrame(
    data = confusion_matrix(y_test, grid_search_pipeline.predict(x_test)),
    index = ["True 0", "True 1"],
    columns = ["Predicted 0", "Predicted 1"],
)

metrics = {
    "type":"cm_matrix",
    "dataset":"test",
    "true_0": {
        "predicted_0": int(cm_test.loc["True 0", "Predicted 0"]),
        "predicted_1": int(cm_test.loc["True 0", "Predicted 1"]),
    },
    "true_1": {
        "predicted_0": int(cm_test.loc["True 1", "Predicted 0"]),
        "predicted_1": int(cm_test.loc["True 1", "Predicted 1"]),        
    },
}


with open("../files/output/metrics.json", "a") as file:
    file.write(str(metrics).replace("'", '"'))
    file.write("\n")

display(metrics)