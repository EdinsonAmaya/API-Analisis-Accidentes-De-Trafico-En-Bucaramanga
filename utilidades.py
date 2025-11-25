import pandas as pd
import os

def cargar_y_limpiar_datos():
    
    # ... (código de ruta) ...
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'AccidentesBucaramanga.xlsx')
    sheet_name = 'hoja 1'
    
    df = None # Inicializa df para el bloque except

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        df.columns = df.columns.str.strip() 

        column_mapping = {
            'AÃ‘O': 'AÑO',    # Ejemplo de corrección de AÑO
            'DÃ\x8dA': 'DÍA',    # **¡AJUSTA ESTO!** Si en el print sale 'DÃA' (sin espacio), quita el espacio de 'DÃ A'
            'DÍA ': 'DÍA',     # Para corregir si tiene un espacio al final
            'Ã\xad': 'í',
        }
        df.rename(columns=column_mapping, inplace=True)
                
        # Convertir FECHA a datetime
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

        # Extraer solo la hora
        df['HORA'] = pd.to_datetime(df['HORA'], errors='coerce').dt.time
        df['HORA_INT'] = df['HORA'].apply(lambda x: x.hour if pd.notnull(x) else None)

        # Crear mes numérico
        # Se asume que la columna MES existe y tiene el nombre correcto.
        df['MES_NUM'] = df['MES'].str.split('.').str[0].astype(int, errors='ignore')

        # Crear día de la semana numérico
        dia_map = {
            'Lunes': 1, 'Martes': 2, 'Miercoles': 3, 'Jueves': 4,
            'Viernes': 5, 'Sabado': 6, 'Domingo': 7
        }
        
        # Usar la lógica de split solo si la columna DÍA tiene el formato '1. Lunes'
        df['DIA_NUM'] = df['DÍA'].str.split('.').str[1].str.strip().map(dia_map)
        
        return df

    except Exception as e:
        print(f"Error cargando datos: {e}")
        # El bloque de registro de errores es útil, lo mantendremos.
        if df is not None:
             print("Las columnas disponibles al fallar eran:")
             print(df.columns.tolist())

        return None