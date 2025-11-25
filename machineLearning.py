from flask import jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from utilidades import cargar_y_limpiar_datos

# Modelos de machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

#---------------------------------------------------------

def preparar_datos_ml(df):
    """Prepara los datos para modelos de machine learning"""

    # Seleccionar variables importantes basadas en el análisis exploratorio
    features = df[['HORA_INT', 'MES_NUM', 'DIA_NUM', 'AÑO', 'DIURNIO/NOCTURNO',
                   'PEATON', 'AUTOMOVIL', 'MOTO', 'BICICLETA']].copy()

    # Target: Gravedad del accidente (si está disponible)
    # Si no existe, crear una variable objetivo basada en características
    if 'GRAVEDAD' in df.columns:
        target = df['GRAVEDAD']
    else:
        # Crear variable objetivo sintética basada en vehículos involucrados
        # Considerar accidentes graves si involucran peatones o múltiples vehículos
        conditions = [
            (df['PEATON'] > 0) |
            ((df['AUTOMOVIL'] + df['MOTO'] + df['BICICLETA']) > 2)
        ]
        choices = ['GRAVE']
        target = np.select(conditions, choices, default='LEVE')
        print("Variable objetivo creada sintéticamente")

    # Limpiar datos faltantes
    features = features.fillna(method='ffill').fillna(method='bfill')

    # Codificar variables categóricas
    le = LabelEncoder()
    if 'DIURNIO/NOCTURNO' in features.columns:
        features['DIURNIO_NOCTURNO_ENC'] = le.fit_transform(features['DIURNIO/NOCTURNO'])
        features = features.drop('DIURNIO/NOCTURNO', axis=1)

    # Codificar target si es categórico
    if target.dtype == 'object':
        target_encoded = le.fit_transform(target)
        target_classes = le.classes_
    else:
        target_encoded = target
        target_classes = None

    return features, target_encoded, target_classes

def entrenar_modelos(X, y):
    """Entrena y evalúa los tres modelos de machine learning"""

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Escalar características para KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelos = {}
    resultados = {}

    # 1. K-Nearest Neighbors (KNN)
    print("=== ENTRENANDO K-NEAREST NEIGHBORS ===")
    knn = KNeighborsClassifier()

    # Búsqueda de hiperparámetros
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
    grid_knn.fit(X_train_scaled, y_train)

    mejores_params_knn = grid_knn.best_params_
    knn_mejor = grid_knn.best_estimator_

    # Predicciones
    y_pred_knn = knn_mejor.predict(X_test_scaled)

    modelos['KNN'] = {
        'modelo': knn_mejor,
        'scaler': scaler,
        'predicciones': y_pred_knn,
        'mejores_params': mejores_params_knn
    }

    # 2. Árbol de Decisión
    print("\n=== ENTRENANDO ÁRBOL DE DECISIÓN ===")
    arbol = DecisionTreeClassifier(random_state=42)

    param_grid_arbol = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid_arbol = GridSearchCV(arbol, param_grid_arbol, cv=5, scoring='accuracy', n_jobs=-1)
    grid_arbol.fit(X_train, y_train)

    mejores_params_arbol = grid_arbol.best_params_
    arbol_mejor = grid_arbol.best_estimator_

    y_pred_arbol = arbol_mejor.predict(X_test)

    modelos['Arbol_Decision'] = {
        'modelo': arbol_mejor,
        'predicciones': y_pred_arbol,
        'mejores_params': mejores_params_arbol
    }

    # 3. Naive Bayes
    print("\n=== ENTRENANDO NAIVE BAYES ===")
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)

    modelos['Naive_Bayes'] = {
        'modelo': nb,
        'predicciones': y_pred_nb,
        'mejores_params': 'default'
    }

    # Evaluar modelos
    for nombre, modelo_info in modelos.items():
        y_pred = modelo_info['predicciones']
        accuracy = accuracy_score(y_test, y_pred)

        resultados[nombre] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"\n--- RESULTADOS {nombre} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Mejores parámetros: {modelo_info['mejores_params']}")
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))

    return modelos, resultados, X_test, y_test

def visualizar_resultados(modelos, resultados, target_classes, X, y):
    """Visualiza los resultados de los modelos"""

    # Comparación de accuracy
    nombres = list(resultados.keys())
    accuracies = [resultados[nombre]['accuracy'] for nombre in nombres]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(nombres, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Comparación de Accuracy entre Modelos')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Añadir valores en las barras
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')

    plt.show()

    # Matrices de confusión
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (nombre, resultado) in enumerate(resultados.items()):
        cm = resultado['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'Matriz de Confusión - {nombre}')
        if target_classes is not None:
            axes[idx].set_xticklabels(target_classes)
            axes[idx].set_yticklabels(target_classes)

    plt.tight_layout()
    plt.show()

    # Importancia de características (solo para árbol de decisión)
    if 'Arbol_Decision' in modelos:
        importancia = modelos['Arbol_Decision']['modelo'].feature_importances_
        caracteristicas = X.columns

        plt.figure(figsize=(10, 6))
        indices = np.argsort(importancia)[::-1]
        plt.bar(range(len(importancia)), importancia[indices])
        plt.xticks(range(len(importancia)), [caracteristicas[i] for i in indices], rotation=45)
        plt.title('Importancia de Características - Árbol de Decisión')
        plt.tight_layout()
        plt.show()

def analizar_patrones(modelos, df, features):
    """Analiza patrones importantes encontrados por los modelos"""

    print("\n=== ANÁLISIS DE PATRONES IMPORTANTES ===")

    # Patrones temporales
    print("\n1. PATRONES TEMPORALES:")
    print(f"- Hora pico de accidentes: {df['HORA_INT'].mode()[0]} horas")
    print(f"- Día con más accidentes: {df['DIA_NUM'].mode()[0]}")
    print(f"- Mes con más accidentes: {df['MES_NUM'].mode()[0]}")

    # Vehículos más involucrados
    vehiculos = ['PEATON', 'AUTOMOVIL', 'MOTO', 'BICICLETA']
    total_vehiculos = df[vehiculos].sum()
    print("\n2. VEHÍCULOS MÁS INVOLUCRADOS:")
    for vehiculo, total in total_vehiculos.sort_values(ascending=False).items():
        print(f"- {vehiculo}: {total} accidentes")

    # Patrones de gravedad si existe la variable
    if 'GRAVEDAD' in df.columns:
        print("\n3. DISTRIBUCIÓN DE GRAVEDAD:")
        print(df['GRAVEDAD'].value_counts())


"""Función principal"""
print("=== ANÁLISIS DE ACCIDENTES CON MACHINE LEARNING ===\n")

# Cargar datos
df = cargar_y_preparar_datos()
if df is None:
    print("No se pudieron cargar los datos. Terminando el análisis.")
    exit()

# Entrenar modelos
modelos, resultados, X_test, y_test = entrenar_modelos(features, target)

# Visualizar resultados
visualizar_resultados(modelos, resultados, target_classes, features, target)

# Analizar patrones
analizar_patrones(modelos, df, features)

print("\n=== ANÁLISIS COMPLETADO ===")



#---------------------------------------------------------

def mostrarCaracteristicas():
    data = {
    'Caracteristicas seleccionadas': list(features.columns),
    'Tamaño de características': features.shape,
    'Clases_target': target_classes
    }

    return jsonify(data)
