from flask import jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from utilidades import cargar_y_limpiar_datos

# Cargar datos usando la función de utilidad
df = cargar_y_limpiar_datos()

def mostrarInformacion():

    if df is None:
        return "<h1>Error de Datos</h1><p>No se pudo cargar el archivo de datos.</p>"
    
    # Cálculo de métricas clave
    total_registros = len(df)
    columnas = df.columns.tolist()
    top_barrios = df['BARRIO'].value_counts().head(5).reset_index()
    top_barrios.columns = ['barrio', 'conteo']
    top_barrios_list = top_barrios.to_dict('records')

    data = {
        'total_registros': total_registros,
        'columnas': columnas,
        'top_barrios': top_barrios_list
    }
    return jsonify(data)

def mostrarGraficaPorAnio():
     # Accidentes por año
    accidents_per_year = df['AÑO'].value_counts().sort_index()
    json_data = accidents_per_year.to_json(orient='index')
    return json_data

def mostrarGraficaGravedad():
     # Accidentes por año
    severity_dist = df['GRAVEDAD'].value_counts()
    print(severity_dist)
    json_data = severity_dist.to_json(orient='index')
    return json_data

def mostrarGraficaCorrelacion():

    vehicle_cols = ['PEATON', 'AUTOMOVIL', 'CAMPERO', 'CAMIONETA', 'MICRO', 'BUSETA', 'BUS', 'CAMION', 'VOLQUETA', 'MOTO', 'BICICLETA', 'OTRO']
    # Correlación entre variables numéricas (vehículos)
    corr_matrix = df[vehicle_cols + ['AÑO', 'MES_NUM', 'DIA_NUM']].corr()
    json_data = corr_matrix.to_json(orient='index')
    return json_data