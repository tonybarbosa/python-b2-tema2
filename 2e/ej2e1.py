"""
Enunciado:
Desarrolla un conjunto de funciones para realizar análisis y visualización de datos sobre el conjunto de datos Iris,
utilizando Pandas y Matplotlib. El objetivo es explorar las características de las especies de iris mediante gráficos.

diccionario_colores = {0: 'green', 1: 'red', 2: 'blue'}.

Funciones a desarrollar:
    
- plot_area_graph(df, column_name, ax=None) -> None
    Descripción: Genera un gráfico de área para visualizar la distribución de un atributo específico entre las diferentes
    especies de iris. El eje X representa el índice del DataFrame, y el eje Y representa el valor del atributo especificado
    por column_name. Las áreas son coloreadas según diccionario_colores.
    Parámetros:
    df: DataFrame que contiene los datos.
    column_name: Nombre de la columna cuyos datos se quieren visualizar en el gráfico de área.

- plot_scatter_graph(df, column_name_x, column_name_y, ax=None) -> None
    Descripción: Crea un gráfico de dispersión para comparar dos atributos entre las diferentes especies de iris. Los ejes
    representan los valores de los atributos especificados por column_name_x y column_name_y, con los puntos coloreados según
    diccionario_colores.
    Parámetros:
    df: DataFrame que contiene los datos.
    column_name_x: Nombre de la columna que se quiere visualizar en el eje X del gráfico de dispersión.
    column_name_y: Nombre de la columna que se quiere visualizar en el eje Y del gráfico de dispersión.

Ejemplo:
    plot_area_graph(dataframe, 'petal length (cm)')
    plot_scatter_graph(dataframe, 'sepal length (cm)', 'sepal width (cm)')

Salida esperada:
- Un gráfico de área que muestre la distribución de la longitud del pétalo para las diferentes especies de iris.
- Un gráfico de dispersión que compare la longitud y el ancho del sépalo entre las diferentes especies de iris.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_area_graph(df, column_name, ax=None):
    # Write here your code
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    color_map = {0: "green", 1: "red", 2: "blue"}

    for target in df["target"].unique():
        subset = df[df["target"] == target]
        ax.fill_between(
            subset.index,
            subset[column_name],
            label=f"Target {target}",
            color=color_map[target],
            alpha=0.5,
        )

    ax.legend()
    ax.set_title(f"Area Graph of {column_name}")
    ax.set_xlabel("Index")
    ax.set_ylabel(column_name)
    return fig, ax



def plot_scatter_graph(df, column_name_x, column_name_y, ax=None):
    # Write here your code
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    color_map = {0: "green", 1: "red", 2: "blue"}
    colors = df["target"].map(color_map)

    ax.scatter(df[column_name_x], df[column_name_y], c=colors, label="Targets")
    legends = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=f"Target {i}",
        )
        for i, color in color_map.items()
    ]
    ax.legend(handles=legends)

    ax.set_title(f"Scatter Plot of {column_name_x} vs {column_name_y}")
    ax.set_xlabel(column_name_x)
    ax.set_ylabel(column_name_y)
    return fig, ax


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    path_csv = current_dir / "data/iris_dataset.csv"
    dataframe = pd.read_csv(path_csv)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_area_graph(dataframe, "petal length (cm)", ax=axs[0])
    plot_scatter_graph(dataframe, "sepal length (cm)", "sepal width (cm)", ax=axs[1])

    plt.tight_layout()
    plt.show()
