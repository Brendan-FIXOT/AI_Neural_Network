import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self):
        self.results = []  # Stocke 1 si réussi, 0 si échec

    def add_result(self, success):
        """Ajoute un résultat (True pour réussi, False pour échec)"""
        self.results.append(1 if success else 0)

    def plot_results(self):
        """Affiche un graphique des performances au fil des essais"""
        if not self.results:
            print("Aucun résultat à afficher.")
            return
        
        essais = np.arange(1, len(self.results) + 1)
        taux_reussite_cumulatif = np.cumsum(self.results) / essais  # Moyenne cumulative

        plt.figure(figsize=(10, 5))
        plt.plot(essais, taux_reussite_cumulatif, marker='o', linestyle='-', color='b', label="Taux de réussite")
        plt.xlabel("Nombre d'essais")
        plt.ylabel("Taux de réussite cumulé")
        plt.title("Évolution du taux de réussite lors de l'entraînement")
        plt.legend()
        plt.grid()
        plt.show()
