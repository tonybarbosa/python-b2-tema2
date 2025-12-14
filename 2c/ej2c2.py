"""
Enunciado:
Desarrolla un conjunto de funciones para realizar análisis estadístico sobre un conjunto de datos de calificaciones de
estudiantes utilizando Pandas. El objetivo es proporcionar una visión comprensiva del rendimiento de los estudiantes a
través del análisis de sus calificaciones almacenadas en un archivo CSV. Para ello, se compararán dos enfoques de
análisis descriptivo: uno mediante una función personalizada y otro utilizando la función describe de Pandas.

Las funciones a desarrollar son:
- Leer un archivo CSV y convertirlo en un DataFrame:
    read_csv_basic(file_path: str) -> pd.DataFrame
- Realizar una descripción estadística personalizada del DataFrame:
    custom_dataframe_describe(df: pd.DataFrame) -> pd.DataFrame
    Esta función debe calcular la media, mediana, desviación estándar, mínimo, máximo, y los cuartiles del DataFrame
    proporcionado.
- Utilizar la función describe de Pandas para obtener estadísticas descriptivas del DataFrame:
    pandas_dataframe_describe(df: pd.DataFrame) -> pd.DataFrame

Parámetros:
- file_path (str): Ruta del archivo CSV que contiene los datos de calificaciones.
- df (pd.DataFrame): DataFrame de pandas que contiene los datos.

Ejemplo:
    archivo_csv_path = 'data/grades.csv'
    dataframe = read_csv_basic(archivo_csv_path)

    print("Custom DataFrame:")
    print(custom_dataframe_describe(dataframe))

    print("\nDataFrame with Pandas:")
    print(pandas_dataframe_describe(dataframe))

Salida esperada:
- DataFrame que muestra la descripción estadística realizada manualmente de las calificaciones, incluyendo la media,
mediana, desviación estándar, mínimo, máximo, y los cuartiles para cada columna del conjunto de datos.
- DataFrame generado por la función describe de Pandas, que proporciona un resumen estadístico similar al anterior,
facilitando la comparación entre ambos enfoques.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def read_csv_basic(file_path: str) -> pd.DataFrame:
    # Write here your code
    df = pd.read_csv(file_path)
    return df


def custom_dataframe_describe(df: pd.DataFrame) -> pd.DataFrame:
    # Write here your code
    df_numeric = df.select_dtypes(include=[np.number])  # crea el dataframe solo con las columnas numericas
    descriptions = pd.DataFrame(  # crea un dataframe con las estadisticas calculadas de forma personalizada
        {
            "count": df_numeric.count(),
            "mean": df_numeric.mean(),
            "median": df_numeric.median(),
            "std": df_numeric.std(),
            "min": df_numeric.min(),
            "25%": df_numeric.quantile(0.25),
            "50%": df_numeric.quantile(0.5),
            "75%": df_numeric.quantile(0.75),
            "max": df_numeric.max(),
        }
    )
    return descriptions.T


def pandas_dataframe_describe(df: pd.DataFrame) -> pd.DataFrame:
    # Write here your code
     return df.describe()  # Utiliza la función describe de Pandas y da los datros estadisticos basicos


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    FILE_PATH = current_dir / "data/grades.csv"
    dataframe = read_csv_basic(FILE_PATH)
    print("DataFrame Original:\n", dataframe.head(), end="\n\n")   

    print("Custom Describe of the DataFrame:")
    print(custom_dataframe_describe(dataframe), end="\n\n")

    print("Pandas Describe of the DataFrame:")
    print(pandas_dataframe_describe(dataframe))
