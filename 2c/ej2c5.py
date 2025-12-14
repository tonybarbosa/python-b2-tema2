"""
Enunciado:
Desarrolla un conjunto de funciones en Python para manejar y procesar un conjunto de datos de calificaciones que 
potencialmente contiene valores faltantes o no válidos. El conjunto de datos está almacenado en un archivo CSV y
contiene calificaciones en diversas materias para un grupo de estudiantes. El objetivo es limpiar los datos, eliminar
filas con datos faltantes en columnas clave, y aplicar técnicas de imputación para tratar los valores faltantes en
ciertas columnas.

Las funciones a desarrollar son:
1. Leer el archivo CSV: `read_csv(filepath: str)` que lee el archivo CSV desde la ruta especificada y devuelve un 
DataFrame de pandas.
2. Limpiar el DataFrame: `clean_dataframe(df)` que limpia el DataFrame reemplazando valores no válidos ('Null', '-',
'NA', 'na', y espacios en blanco) por `NaN`, y convierte las columnas numéricas a su tipo correspondiente, excepto la
primera columna.
3. Eliminar filas con `NaN` en una columna específica: `dropna_specific_row_in_column(df, column_name: str)` que
elimina las filas que contienen un valor `NaN` en la columna especificada por `column_name`.
4. Rellenar valores faltantes en una columna específica: `fillna_method(df, column_name: str, fill_method='ffill',
fill_value=None, limit=1)` que rellena los valores faltantes en la columna especificada utilizando un método de 
relleno especificado (`ffill` para relleno hacia adelante con un límite de 1, o `mean` para rellenar con el promedio
de la columna).

Parámetros:
- filepath (str): Ruta al archivo CSV que contiene los datos de calificaciones.
- column_name (str): Nombre de la columna en la cual aplicar la eliminación de filas o el relleno de valores 
faltantes.
- fill_method (str): Método de relleno para tratar valores faltantes (`ffill` para relleno hacia adelante, `mean` 
para el promedio).
- fill_value (Optional[float]): Valor específico de relleno a usar si se especifica. Útil solo cuando 
`fill_method` es `mean`.
- limit (int)`: Número máximo de rellenos consecutivos a aplicar cuando se utiliza `ffill`.

Ejemplo de uso:
    df_cleaned = clean_dataframe(df_v2)
    df_drop_na_rows = dropna_specific_row_in_column(df_cleaned, 'Name')
    df_filled_column_ffill = fillna_method(df_drop_na_rows, 'Hindi', fill_method='ffill', limit=1)
    df_filled_column_mean = fillna_method(df_filled_column_ffill, 'Maths', fill_method='mean')


Salida esperada:
- Un DataFrame limpio y procesado donde las filas con valores faltantes en columnas clave han sido eliminadas, y los
valores faltantes en ciertas columnas han sido tratados mediante técnicas de imputación específicas.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def read_csv(filepath):
    # Write here your code
    df = pd.read_csv(filepath)
    return df


def clean_dataframe(df):
    # Write here your code
    
    '''
    # es mi codigoy da error F also en el test
    valores_a_reemplazar = ['Null', '-', 'NA', 'na', 'nan' ' ']
    df_limpio = df.replace(valores_a_reemplazar, np.nan)
    columnas=["Hindi","English","Science","Maths","History","Geography"]
    df_limpio[columnas] = df_limpio[columnas].apply(pd.to_numeric, errors='coerce').astype(float)
    return df_limpio
    '''
    
    # con este codigo tampoco pasa el testy continua con el error F

    
    df.replace(["Null", "-", "NA", "na", "nan", ""], np.nan, inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x).replace(
        "", np.nan
    )

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df




def dropna_specific_row_in_column(df, column_name):
    # Write here your code
    df1 = df.dropna(subset=["Name"])
    return df1


def fillna_method(df, column_name, fill_method="ffill", fill_value=None, limit=1):
    # Write here your code
    if fill_method == "ffill":
        df[column_name] = df[column_name].ffill(limit=limit)
    elif fill_method == "mean":
        #df[column_name] = pd.to_numeric(df[column_name], errors="coerce") #los valores parecn numericos pero son alfanumericos
        if fill_value is not None:
            mean_value = fill_value
        else:
            mean_value = df[column_name].mean()
        df[column_name] = df[column_name].fillna(mean_value)
    else:
        raise ValueError("fill_method debe ser 'ffill' o 'mean'")
    return df

# Para probar el código, descomenta las siguientes líneas y asegúrate de que el path al archivo sea correcto

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    FILE_PATH = current_dir / "data/grades_na.csv"
    dataframe = read_csv(FILE_PATH)
    df_cleaned = clean_dataframe(dataframe)
    df_drop_na_rows = dropna_specific_row_in_column(df_cleaned, "Name")
    print(df_drop_na_rows)   
    df_filled_column_ffill = fillna_method(
        df_drop_na_rows, "Hindi", fill_method="ffill", limit=1
    )
    
    df_filled_column_mean = fillna_method(
        df_filled_column_ffill, "Maths", fill_method="mean"
    )

    print(dataframe.head())
    print(df_filled_column_mean.head())
