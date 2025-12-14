"""
Enunciado:
Desarrolla un conjunto de funciones para realizar y visualizar un análisis de regresión lineal entre el número promedio
de habitaciones por vivienda (RM) y el valor medio de las viviendas ocupadas por sus propietarios (MEDV) en el conjunto
de datos de viviendas, utilizando `scipy.stats.linregress` y `matplotlib` para la visualización. El archivo CSV
'housing.csv' contiene datos relevantes para este análisis.

Las funciones a desarrollar son:
1. Realizar análisis de regresión lineal: `perform_linear_regression(data: pd.Dataframe, variable_1: str, variable_2: str)
que lee un archivo CSV y realiza una regresión lineal entre dos variables dadas, devolviendo la pendiente, la
intersección, el valor r, el valor p y el error estándar de la pendiente.
2. Graficar la regresión lineal y los puntos de datos: `plot_regression_line(data: pd.Dataframe, variable_1: str,
variable_2: str, slope: float, intercept: float)` que utiliza `matplotlib` para visualizar la línea de regresión lineal
junto con los puntos de datos de las dos variables seleccionadas.

Parámetros:
    data ( pd.Dataframe): Pandas dataframe que contiene los datos de vivienda.
    variable_1 (str): Nombre de la primera variable para el análisis de regresión (e.g., 'RM').
    variable_2 (str): Nombre de la segunda variable para el análisis de regresión (e.g., 'MEDV').
    slope (float): Pendiente de la línea de regresión lineal.
    intercept (float): Intersección de la línea de regresión lineal con el eje y.

Ejemplo de uso:
    slope, intercept, r_value, p_value, std_err = perform_linear_regression(data, 'RM', 'MEDV')
    plot_regression_line(data, 'RM', 'MEDV', slope, intercept)

Salida esperada:
    Una gráfica que muestra tanto los puntos de datos de RM vs MEDV como la línea de regresión lineal calculada.
"""

from pathlib import Path
import pandas as pd
from scipy.stats import linregress
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def perform_linear_regression(
    data: pd.DataFrame, variable_1: str, variable_2: str
) -> Tuple[float, float, float, float, float]:
    # Write here your code
    x = data[variable_1]
    y = data[variable_2]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err


def plot_regression_line(
    data: pd.DataFrame,
    variable_1: str,
    variable_2: str,
    slope: float,
    intercept: float,
    return_fig_ax_test=False,
):
    # Write here your code
    #pass
    x = data[variable_1]
    y = data[variable_2]
    # --- 2. Crear la figura y los ejes ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 3. Graficar los puntos de datos originales ---
    # RM en el eje X (variable independiente), MEDV en el eje Y (variable dependiente)
    plt.scatter(x, y, color='blue', label='Datos Originales (RM vs MEDV)', alpha=0.6)

    # --- 4. Graficar la línea de regresión ---
    plt.plot(x, slope * x + intercept, color='red', 
            label=f'Línea de Regresión: MEDV = {slope:.2f}*RM + {intercept:.2f}')

    # --- 5. Añadir etiquetas, título y leyenda ---
    plt.title('Linear Regression between RM and MEDV')  # Regresión Lineal entre Número Medio de Habitaciones (RM) y Valor Medio (MEDV)
    plt.xlabel('RM')  # RM (Número Medio de Habitaciones)
    plt.ylabel('MEDV')  # MEDV (Valor Medio de la Propiedad en $1000s)
    plt.legend()
    plt.grid(True) # Opcional: añade una cuadrícula
    plt.show() # Mostrar el gráfico
    return fig, ax


# Para probar el código, descomenta este código
if __name__ == '__main__':
    current_dir = Path(__file__).parent
    HOUSING_CSV_PATH = current_dir / 'data/housing.csv'
    variable_1 = 'RM'
    variable_2 = 'MEDV'
    data = pd.read_csv(HOUSING_CSV_PATH, skiprows=14)
    perform_linear_regression(data, variable_1, variable_2)
    slope, intercept, r_value, p_value, std_err = perform_linear_regression(data, variable_1, variable_2)

    print(f'Análisis de Regresión Lineal entre {variable_1} y {variable_2}:')
    print(f'Pendiente: {slope}, Intersección: {intercept}, Valor r: {r_value}, Valor p: {p_value},'
        f'Error estándar: {std_err}')
    print(f"\nEcuación de Regresión: MEDV = {slope:.4f} * RM + {intercept:.4f}")

    # Graficar la línea de regresión
    fig, ax = plot_regression_line(data, variable_1, variable_2, slope, intercept, return_fig_ax_test=False)


'''
Explicación de los Resultados
La función scipy.stats.linregress(x, y) te devuelve los siguientes parámetros:
slope (Pendiente): Indica cuánto cambia la variable dependiente (MEDV) por cada unidad de cambio en la variable independiente (RM).
intercept (Intercepción): Es el valor predicho de MEDV cuando RM es 0.
r_value (Coeficiente de Correlación): Mide la fuerza y dirección de la relación lineal. Los valores están entre -1 y 1.
r_value**2 (R-cuadrado, Coeficiente de Determinación): Es la proporción de la varianza en MEDV que es predecible a partir de RM.
p_value (Valor p): Indica la probabilidad de que la relación observada haya ocurrido por casualidad. Un valor pequeño (típicamente $< 0.05$) sugiere una relación significativa.
std_err (Error Estándar de la Pendiente): Mide la precisión de la estimación de la pendiente.

'''