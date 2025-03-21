import numpy as np
from train import Train
from model import NeuralNetwork
from data import iris_data, iris_target
from train_visualization import TrainingVisualizer

def main() :
    # Affichage des données importées
    print("Données Iris chargées :", iris_data.shape)
    print("Target Iris charhées : ", iris_target.shape)
    
    input_size=4
    output_size=3

    # Création du réseau (pour stocker les poids pendant les entraînements)
    nn = NeuralNetwork(input_size, output_size)
    
    numbercycle = 500
    
    visualizer = TrainingVisualizer()
    
    for i in range(numbercycle) :
        trainer = Train(nn, iris_data, iris_target, visualizer)
        trainer.train()
        
    visualizer.plot_results()
    
if __name__ == "__main__" :
    print("Exécution de main.py")
    main()