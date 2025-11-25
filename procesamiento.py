import pandas as pd
import numpy as np
import json
from datetime import datetime
from flask import Response
from utilidades import cargar_y_limpiar_datos

# Modelos de machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class AccidentesMLAPI:
    
    def __init__(self):
        """Inicializa la clase cargando y preparando los datos."""
        self.df = None
        self.features = None
        self.target_encoded = None
        self.target_classes = None
        self.modelos = {}
        self.resultados = {}
        
        # Cargar y preparar el DataFrame principal

        df = cargar_y_limpiar_datos()

        self.df = df

        features = df[['HORA_INT', 'MES_NUM', 'DIA_NUM', 'AÑO', 'DIURNIO/NOCTURNO',
                       'PEATON', 'AUTOMOVIL', 'MOTO', 'BICICLETA']].copy()

        # Target: Gravedad del accidente
        if 'GRAVEDAD' in df.columns:
            target = df['GRAVEDAD']
        else:
            # Crear variable objetivo sintética
            conditions = [(df['PEATON'] > 0) | ((df['AUTOMOVIL'] + df['MOTO'] + df['BICICLETA']) > 2)]
            choices = ['GRAVE']
            target = np.select(conditions, choices, default='LEVE')

        # Limpiar datos faltantes
        features = features.fillna(method='ffill').fillna(method='bfill')

        # Codificar variables categóricas
        le = LabelEncoder()
        if 'DIURNIO/NOCTURNO' in features.columns:
            features['DIURNIO_NOCTURNO_ENC'] = le.fit_transform(features['DIURNIO/NOCTURNO'])
            features = features.drop('DIURNIO/NOCTURNO', axis=1)

        # Codificar target
        if target.dtype == 'object' or target.dtype.kind in np.typecodes['AllInteger']:
            target_encoded = le.fit_transform(target)
            target_classes = le.classes_.tolist()
        else:
            target_encoded = target
            target_classes = None

        self.features = features
        self.target_encoded = target_encoded
        self.target_classes = target_classes

    # --- Funciones de Retorno JSON para API ---

    def get_info_inicial_json(self):
        """
        Retorna la información básica del dataset cargado en formato JSON.
        Endpoint: /api/info
        """
        if self.df is None or self.df.empty:
            return json.dumps({
                "status": "error",
                "message": "Error al cargar o preparar los datos."
            })
            
        data = {
            "status": "ok",
            "dimensiones": list(self.df.shape),
            "columnas_disponibles": self.df.columns.tolist(),
            "caracteristicas_ml": self.features.columns.tolist(),
            "target_classes": self.target_classes,
            "distribucion_target": pd.Series(self.target_encoded).value_counts().to_dict()
        }
        return json.dumps(data)

    def entrenar_modelos_json(self):
        """
        Entrena y evalúa los modelos de ML, retornando métricas y parámetros 
        en formato JSON.
        Endpoint: /api/train
        """
        if self.features is None or self.target_encoded is None:
            return json.dumps({
                "status": "error",
                "message": "Los datos para Machine Learning no están listos. Revise la carga."
            })

        X = self.features
        y = self.target_encoded

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        modelos_temp = {}
        resultados_temp = {}

        # 1. K-Nearest Neighbors (KNN) - Escalado de datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        param_grid_knn = {'n_neighbors': [5, 7, 9], 'weights': ['distance']}
        grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3, scoring='accuracy', n_jobs=-1)
        grid_knn.fit(X_train_scaled, y_train)
        y_pred_knn = grid_knn.best_estimator_.predict(X_test_scaled)

        modelos_temp['KNN'] = grid_knn.best_estimator_
        resultados_temp['KNN'] = self._obtener_metricas(y_test, y_pred_knn, grid_knn.best_params_)

        # 2. Árbol de Decisión
        param_grid_arbol = {'max_depth': [5, 7, 10], 'criterion': ['gini']}
        grid_arbol = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_arbol, cv=3, scoring='accuracy', n_jobs=-1)
        grid_arbol.fit(X_train, y_train)
        y_pred_arbol = grid_arbol.best_estimator_.predict(X_test)

        modelos_temp['Arbol_Decision'] = grid_arbol.best_estimator_
        resultados_temp['Arbol_Decision'] = self._obtener_metricas(y_test, y_pred_arbol, grid_arbol.best_params_)

        # 3. Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        
        modelos_temp['Naive_Bayes'] = nb
        resultados_temp['Naive_Bayes'] = self._obtener_metricas(y_test, y_pred_nb, 'default')

        # Almacenar modelos y resultados (sin el objeto modelo ni la matriz)
        self.modelos = {k: v for k, v in modelos_temp.items()} # Guardamos para predicciones futuras
        self.resultados = resultados_temp
        
        # Eliminar las matrices de confusión que no son directamente JSON-serializables para el retorno
        resultados_para_json = self._limpiar_resultados_para_json(resultados_temp)

        return json.dumps({
            "status": "ok",
            "resultados_entrenamiento": resultados_para_json,
            "clases": self.target_classes
        })

    def _obtener_metricas(self, y_true, y_pred, best_params):
        """Calcula métricas clave."""
        reporte = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Simplificar el reporte para JSON
        reporte_json = {
            k: {vk: vv for vk, vv in v.items() if isinstance(vv, (float, int))} 
            for k, v in reporte.items() if isinstance(v, dict)
        }
        reporte_json['accuracy'] = accuracy_score(y_true, y_pred)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': reporte_json,
            'confusion_matrix': cm,
            'mejores_params': best_params
        }

    def _limpiar_resultados_para_json(self, resultados):
        """Elimina objetos complejos de los resultados para la serialización final."""
        resultados_limpios = {}
        for nombre, res in resultados.items():
            # Copiamos para no modificar el objeto original
            res_copia = res.copy()
            # Convertimos la matriz de numpy a lista de Python para JSON
            res_copia['confusion_matrix'] = res_copia['confusion_matrix']
            # Dejamos solo los parámetros serializables
            resultados_limpios[nombre] = res_copia
            
        return resultados_limpios

    def analizar_patrones_json(self):
        """
        Retorna patrones de análisis exploratorio de datos (EDA) en formato JSON.
        Endpoint: /api/patterns
        """
        if self.df is None or self.df.empty:
            return json.dumps({
                "status": "error",
                "message": "Los datos no están cargados. Revise la carga."
            })

        df = self.df
        
        # 1. Patrones temporales
        patrones_temporales = {
            "hora_pico_accidentes": int(df['HORA_INT'].mode()[0]) if not df['HORA_INT'].empty else None,
            "dia_mas_accidentes": int(df['DIA_NUM'].mode()[0]) if not df['DIA_NUM'].empty else None,
            "mes_mas_accidentes": int(df['MES_NUM'].mode()[0]) if not df['MES_NUM'].empty else None
        }

        # 2. Vehículos más involucrados
        vehiculos = ['PEATON', 'AUTOMOVIL', 'MOTO', 'BICICLETA']
        total_vehiculos = df[vehiculos].sum().sort_values(ascending=False)
        vehiculos_involucrados = total_vehiculos.to_dict()

        # 3. Distribución de gravedad
        distribucion_gravedad = None
        if 'GRAVEDAD' in df.columns:
            distribucion_gravedad = df['GRAVEDAD'].value_counts().to_dict()

        data = {
            "status": "ok",
            "patrones_temporales": patrones_temporales,
            "vehiculos_involucrados": vehiculos_involucrados,
            "distribucion_gravedad": distribucion_gravedad
        }
        return json.dumps(data)