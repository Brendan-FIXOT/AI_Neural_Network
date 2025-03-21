import numpy as np
from sklearn.datasets import load_iris

# Chargement du dataset Iris
iris = load_iris()

# Séparation des datas des target
iris_data = np.array(iris.data)  # Contient uniquement les 4 features
iris_target = np.array(iris.target)  # Contient les labels (0, 1, 2)

# Vérification
if __name__ == "__main__" :
    print("Dataset chargé avec succès !")
    print("iris_data shape:", iris_data.shape)  # (150, 4)
    print("iris_target shape:", iris_target.shape)  # (150,)s