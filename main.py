from flask import Flask, Response
from analisis import mostrarInformacion, mostrarGraficaPorAnio, mostrarGraficaGravedad, mostrarGraficaCorrelacion # Asegúrate que mostrarInformacion está en el nivel superior de analisis.py
from procesamiento import AccidentesMLAPI 
from flask_cors import CORS
import json



app = Flask(__name__)
CORS(app)

try:
    api_handler = AccidentesMLAPI()
    print("Clase AccidentesMLAPI inicializada y datos cargados.")
except Exception as e:
    print(f"Error fatal al inicializar AccidentesMLAPI: {e}")
    api_handler = None
    
# Función auxiliar para manejar la respuesta JSON
def api_response(json_string):
    """Crea una respuesta Flask con Content-Type application/json."""
    return Response(json_string, mimetype='application/json')

@app.route('/informacion')
def informacion():
    return mostrarInformacion()

@app.route('/anios')
def graficaAnios():
    return mostrarGraficaPorAnio()

@app.route('/gravedad')
def graficaGravedad():
    return mostrarGraficaGravedad()

@app.route('/correlacion')
def graficaCorrelacion():
    return mostrarGraficaCorrelacion()

@app.route('/ml/info')
def ml_informacion():
    """Retorna la información inicial del dataset y ML."""
    if api_handler is None:
        return api_response(json.dumps({"status": "error", "message": "API Handler no inicializado."}))
    return api_response(api_handler.get_info_inicial_json())

@app.route('/ml/train')
def ml_entrenamiento():
    """Entrena los modelos y retorna las métricas de evaluación."""
    if api_handler is None:
        return api_response(json.dumps({"status": "error", "message": "API Handler no inicializado."}))
    # Se llama al método para entrenar y obtener el JSON de resultados
    return api_response(api_handler.entrenar_modelos_json())

@app.route('/ml/patterns')
def ml_patrones():
    """Retorna los patrones de análisis exploratorio (EDA)."""
    if api_handler is None:
        return api_response(json.dumps({"status": "error", "message": "API Handler no inicializado."}))
    return api_response(api_handler.analizar_patrones_json())

if __name__ == '__main__':
    # Si la ruta del archivo es relativa, el debug=True puede causar problemas con recarga.
    app.run(debug=True, host='0.0.0.0')