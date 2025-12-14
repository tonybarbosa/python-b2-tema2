"""
Enunciado:
Desarrolla un conjunto de funciones para entrenar un modelo de clasificación y visualizar tanto las fronteras de
decisión como la matriz de confusión, utilizando bibliotecas como Matplotlib, Numpy, y Scikit-learn.

Funciones a desarrollar:
    - create_meshgrid(X, resolution=0.02) -> (np.ndarray, np.ndarray)
     Descripción: 
     Genera un mallado (meshgrid) a partir de los datos de entrada X. Este mallado se utiliza luego para visualizar
     las fronteras de decisión del modelo. 
     Parámetros: 
        X (np.ndarray): Datos de entrada. resolution (float): Resolución del mallado.

    - plot_decision_boundaries(ax, X, y, classifier, resolution=0.02) -> matplotlib.axes._axes.Ax
     Descripción: 
     Utiliza el mallado creado por create_meshgrid y colorea las regiones según las predicciones del clasificador.
     Parámetros: 
        ax (matplotlib.axes._axes.Ax): Eje donde se dibujará.
        X (np.ndarray): Datos de entrada. 
        y (np.ndarray): Etiquetas verdaderas.
        classifier: Modelo clasificador entrenado.
        resolution (float): Resolución del mallado.

    - plot_confusion_matrix(ax, y_true, y_pred, classes) -> matplotlib.axes._axes.Ax
    Descripción:
    Genera y visualiza la matriz de confusión para evaluar el rendimiento del modelo clasificador, mostrando las
    instancias correctamente y erróneamente clasificadas entre las diferentes clases. 
    Parámetros:
        ax (matplotlib.axes._axes.Ax): Eje donde se dibujará. y_true 
        (np.ndarray): Etiquetas verdaderas. y_pred 
        (np.ndarray): Etiquetas predichas. classes 
        (List[str]): Nombres de las clases.

    - train_and_visualize(X, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, classifier)
    Descripción:
    Entrena un clasificador k-NN con los datos de entrenamiento 'X' y 'y', y devuelve los conjuntos de entrenamiento
    y prueba, junto con el clasificador entrenado. El parámetro de 'weights' debe ser 'uniform' y el de 'metric' será
    'minkowski'.
    Parámetros:
        X (np.ndarray): Datos de entrada
        y (np.ndarray): Etiquetas

Ejemplo:
    X_train, X_test, y_train, y_test, classifier = train_and_visualize(X, y)

Salida Esperada:
- Una visualización de las áreas donde el modelo predice cada clase, mostrando la habilidad del clasificador para
separar las clases.
- Una matriz de confusión que resume las predicciones correctas e incorrectas, permitiéndote evaluar cuán bien funciona
el modelo.
"""


#from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def create_meshgrid(X, resolution=0.02):
    # Write here your code
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    print(xx1, xx2)
    return xx1, xx2



def plot_decision_boundaries(ax, X, y, classifier, resolution=0.02):
    # Write here your code
    pass
    xx1, xx2 = create_meshgrid(X, resolution)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap="viridis")

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=20)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    print(ax)
    return ax

def plot_confusion_matrix(ax, y_true, y_pred, classes):
    # Write here your code
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix")
    print(ax)
    return ax

def train_and_visualize(X, y):
    # Write here your code
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    classifier = KNeighborsClassifier(
        n_neighbors=3, weights="uniform", metric="minkowski"
    )
    classifier.fit(X_train, y_train)
    print(X_train, X_test, y_train, y_test, classifier)

    return X_train, X_test, y_train, y_test, classifier


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    X_train, X_test, y_train, y_test, classifier = train_and_visualize(X, y)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundaries(axs[0], X_train, y_train, classifier)
    plot_confusion_matrix(
        axs[1], y_test, classifier.predict(X_test), classes=iris.target_names
    )

    plt.tight_layout()
    plt.show()
