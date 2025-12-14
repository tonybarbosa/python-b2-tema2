"""
Enunciado:
Desarrolla un conjunto de funciones para manipular y transformar un conjunto de datos de productos utilizando pandas. 
Se proporciona un archivo CSV 'products.csv' que contiene datos sobre productos como ID, nombre, marca, categoría, 
precio, calificación, color y tamaño. La tarea es implementar varias funciones para realizar operaciones de apilamiento
(stack), desapilamiento (unstack), pivoteo (pivot), fundido (melt), y transposición (transpose) sobre el DataFrame para
explorar los datos de diferentes maneras.

Las funciones a desarrollar son:

- Leer un archivo CSV y convertirlo en un DataFrame: read_csv_basic(file_path).
- Apilar las filas del DataFrame: stack_dataframe(df), lo que reorganiza el DataFrame para que cada dato sea una fila 
individual, indexada por la información del producto.
- Desapilar el DataFrame apilado: unstack_dataframe(df_stacked), convirtiendo la estructura apilada de vuelta a su 
formato original (o a uno similar).
- Pivotear el DataFrame: pivot_dataframe(df), para reorganizar los datos basados en el ID del producto, las categorías,
y los precios.
- Fundir el DataFrame: melt_dataframe(df), convirtiendo las columnas en filas para obtener un formato "largo" que 
facilite el análisis.
- Transponer el DataFrame: transpose_dataframe(df), para intercambiar filas por columnas y viceversa, proporcionando 
una vista alternativa de los datos.
- Mostrar el DataFrame o su transformación: show_dataframe(nombre, df), para imprimir el nombre de la transformación 
aplicada y visualizar las primeras filas del DataFrame resultante.

Parámetros:
- file_path (str): Ruta del archivo CSV que contiene los datos de los productos.
- df (pd.DataFrame): El DataFrame de pandas que contiene los datos.
- df_stacked (pd.DataFrame): El DataFrame que ha sido previamente apilado.

La implementación de estas funciones permite explorar y analizar el conjunto de datos de productos de múltiples 
maneras, facilitando la identificación de patrones, tendencias, y relaciones entre las diferentes variables.

Ejemplo:
    df_stacked = stack_dataframe(df)
    df_unstacked = unstack_dataframe(df_stacked)
    df_pivoted = pivot_dataframe(df)
    df_melted = melt_dataframe(df)
    df_transposed = transpose_dataframe(df)


Salida esperada:
- Visualizaciones de las primeras filas del DataFrame original y sus transformaciones, mostrando cómo cada operación
altera la estructura de los datos.
"""

from pathlib import Path
import pandas as pd


def read_csv_basic(file_path):
    # Write here your code
    df = pd.read_csv(file_path)
    return df


def stack_dataframe(df):
    # Write here your code
    df1 = df.set_index(
        ["Product ID", "Product Name", "Brand", "Category"]
    ).stack()  # con esta funcion crea un stack pero en cada fila tiene un multiindex
    
    return df1
     
    
def unstack_dataframe(df_stacked):
    # Write here your code
    df1 = df_stacked.unstack()  # con esta funcion crea unstack para revertir el stack
    return df1


def pivot_dataframe(df):
    # Write here your code
    pivot_df = pd.pivot_table(
        df,
        index='Product ID', 
        columns='Category',
        values='Price'
    )
    return pivot_df


def melt_dataframe(df):
    # Write here your code
    df_melted = pd.melt(
        df,
        id_vars=["Product ID", "Product Name", "Brand", "Category", "Color", "Size"],
        value_vars=["Price", "Rating"],
        var_name="Variable",
        value_name="Value"
    )
    return df_melted

def transpose_dataframe(df):
    # Write here your code
    def1= df.transpose()  # con esta funcion transpone el dataframe
    return def1


def show_dataframe(nombre, df):
    # Write here your code
    print(f"### {nombre} DataFrame:")
    print(df)


# # Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    FILE_PATH = current_dir / "data/products.csv"
    df = read_csv_basic(FILE_PATH)
    show_dataframe("BASIC", df)

    df_stacked = stack_dataframe(df)
    show_dataframe("Stack", df_stacked)

    df_unstacked = unstack_dataframe(df_stacked)
    show_dataframe("Unstack", df_unstacked)

    df_pivoted = pivot_dataframe(df)
    show_dataframe("Pivot", df_pivoted)

    df_melted = melt_dataframe(df)
    show_dataframe("Melt", df_melted)

    df_transposed = transpose_dataframe(df)
    show_dataframe("Transpose", df_transposed)
