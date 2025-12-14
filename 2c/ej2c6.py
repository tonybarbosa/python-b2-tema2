"""
Enunciado:
Desarrolla un conjunto de funciones para realizar y visualizar un análisis de componentes principales (PCA) utilizando
`sklearn.decomposition.PCA` en el conjunto de datos de viviendas. Este conjunto de datos contiene varias características
de viviendas en áreas suburbanas de Boston, y el objetivo es reducir la dimensionalidad de los datos para identificar
las principales componentes que explican la mayor parte de la varianza en el conjunto de datos.

Las funciones a desarrollar son:
1. Preparar los datos: `prepare_data_for_pca(file_path: str)` que lee el archivo CSV y prepara los datos eliminando
cualquier característica no numérica. Se debe saltar las primeras 14 filas del archivo CSV, que contienen información
adicional sobre el conjunto de datos.
2. Realizar PCA: `perform_pca(data, n_components: int)` que realiza PCA en los datos preparados, reduciendo a
`n_components` número de dimensiones, en este caso a 4 dimensiones.
3. Visualizar los resultados: `plot_pca_results(pca)` que visualiza los resultados de PCA, incluyendo la varianza
explicada por cada componente principal.

Parámetros:
    file_path (str): Ruta al archivo CSV que contiene los datos del dataset de viviendas.
    n_components (int): Número de componentes principales a retener en el análisis PCA.

Ejemplo de uso:
    pca = perform_pca(data, 4)
    plot_pca_results(pca)

Salida esperada:
    Una visualización de la varianza explicada por los componentes principales y, opcionalmente, una transformación de
    los datos originales proyectada en las principales componentes.
"""

from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def prepare_data_for_pca(file_path: str) -> pd.DataFrame:
    # Write here your code
    df = pd.read_csv(file_path, skiprows=14 )  # Saltar las primeras 14 filas
    #df = df.select_dtypes(include=[float, int])  # Seleccionar solo columnas numéricas
    features = df.drop("MEDV", axis=1, errors="ignore")  # Eliminar la columna (axis=1) MEDV si existe
    return features


def perform_pca(data: pd.DataFrame, n_components: int) -> PCA:
    # Write here your code
    
                # Crea una instancia del objeto StandardScaler.
                # Este objeto calculará la media (μ) y la desviación estándar (σ)
                # para cada columna de tus datos.
    scaler = StandardScaler()
                # Escala los datos para que cada característica tenga media 0 y desviación estándar 1.
                # fit(): El escalador aprende la media y la desviación estándar de cada columna en el DataFrame data.
                # transform(): Aplica la transformación a los datos utilizando la fórmula de estandarización: z=(x - μ) / σ
                # El resultado (data_scaled) es un nuevo conjunto de datos donde cada columna tiene una
                # media de 0 y una desviación estándar de 1.
    data_scaled = scaler.fit_transform(data)
    print("dataescaledªn",data_scaled)
                # Crea una instancia del objeto PCA. El argumento n_components especifica el número de dimensiones
                # (componentes principales) que deseas conservar.
                # Si no se especifica, se usan $N-1$ componentes (donde $N$ es el número de variables originales).
    pca = PCA(n_components=n_components)
                # Ajusta el modelo PCA a los datos escalados.
                # CA aprende cómo transformar los datos. El método realiza lo siguiente:
                # Calcula la matriz de covarianza de los datos escalados.
                # Calcula los autovalores y autovectores de esta matriz.
                # Los autovectores (llamados componentes principales) definen las nuevas direcciones en el espacio de datos
                # que capturan la máxima varianza posible.
    pca.fit(data_scaled)
    print("pca",pca)
    print("ratio",pca.explained_variance_ratio_)
    print("sum",sum(pca.explained_variance_ratio_))
    return pca


def plot_pca_results(pca: PCA) -> tuple:
    # Write here your code
    plt.figure(figsize=(10, 7))
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    plt.bar(
        range(1, pca.n_components_ + 1),
        explained_variance_ratio,
        alpha=0.5,
        align="center",
    )
    plt.step(
        range(1, pca.n_components_ + 1), cumulative_explained_variance, where="mid"
    )
    plt.ylabel("Varianza explicada")
    plt.xlabel("Componentes principales")
    plt.show()
    return explained_variance_ratio, cumulative_explained_variance



# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    FILE_PATH = current_dir / "data/housing.csv"
    dataset = prepare_data_for_pca(FILE_PATH)
    print("dataset\n",dataset.head())
    pca = perform_pca(dataset, 8)
    _, _ = plot_pca_results(pca)
