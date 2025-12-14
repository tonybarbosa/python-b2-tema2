"""
Enunciado:
Explora el análisis de datos mediante la realización de una regresión lineal y la interpolación de un conjunto de datos.
Este ejercicio se centra en el uso de scipy.optimize para llevar a cabo una regresión lineal y en la aplicación de
scipy.interpolate para la interpolación de datos.

Implementa la función linear_regression_and_interpolation(data_x, data_y) que realice lo siguiente:
    - Regresión Lineal: Ajustar una regresión lineal a los datos proporcionados.
    - Interpolación: Crear una interpolación lineal de los mismos datos.

Adicionalmente, implementa la función plot_results(data_x, data_y, results) para graficar los datos originales,
la regresión lineal y la interpolación.

Parámetros:
    - data_x (List[float]): Lista de valores en el eje x.
    - data_y (List[float]): Lista de valores en el eje y correspondientes a data_x.
    - results (Dict): Resultados de la regresión lineal e interpolación.

Ejemplo:
    - Entrada:
        data_x = np.linspace(0, 10, 100)
        data_y = 3 - data_x + 2 + np.random.normal(0, 0.5, 100) # Datos con tendencia lineal y algo de ruido
    - Ejecución:
        results = linear_regression_and_interpolation(data_x, data_y)
        plot_results(data_x, data_y, results)
    - Salida:
        Un gráfico mostrando los datos originales, la regresión lineal y la interpolación.
"""

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import typing as t


def linear_regression_and_interpolation(
    data_x: t.List[float], data_y: t.List[float]
) -> t.Dict[str, t.Any]:
    # Write here your code
    slope, intercept = np.polyfit(data_x, data_y, 1)
    linear_regression = {"slope": slope, "intercept": intercept}

    # Interpolación
    interpolator = interpolate.interp1d(data_x, data_y)
    interpolated_data = interpolator(data_x)

    return {
        "linear_regression": linear_regression,
        "interpolated_data": interpolated_data,
    }




def plot_results(data_x: t.List[float], data_y: t.List[float], results: t.Dict):
    # Write here your code
    plt.figure(figsize=(10, 6))
    plt.scatter(data_x, data_y, label="Datos Originales", color="blue")

    # Regresión lineal
    plt.plot(
        data_x,
        results["linear_regression"]["slope"] * data_x
        + results["linear_regression"]["intercept"],
        label="Regresión Lineal",
        color="red",
    )

    # Interpolación
    plt.plot(
        data_x,
        results["interpolated_data"],
        label="Interpolación",
        color="green",
        linestyle="--",
    )

    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.title("Regresión Lineal e Interpolación de Datos")
    plt.legend()
    plt.show()




# Si quieres probar tu código, descomenta las siguientes líneas y ejecuta el script
data_x = np.linspace(0, 10, 100)
data_y = 3 * data_x + 2 + np.random.normal(0, 2, 100)
results = linear_regression_and_interpolation(data_x, data_y)
plot_results(data_x, data_y, results)


# me faltan conocimientos matematicos para desarrolar estye tipo de analisis
# ejercicio copiado de las soluciones
