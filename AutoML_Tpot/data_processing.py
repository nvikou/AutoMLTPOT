import pandas as pd
import numpy as np

# Fonction de traitement des données,la fonction rertourne X_features, y_target pour l entrainemment 

def data_processing (ws,ws_test):
    """Fonction de traitement des données la fonction rertourne X_features, y_target pour l entrainemment  ."""
#ws = pd.read_excel(file_path, sheet_name="Massive for learning")
#ws_test = pd.read_excel(file_path, sheet_name="Massive")
    # Préparation des données
    L = ws.values.tolist()
    title_L = L[0]
    L = L[1:]  # Correction pour ne pas inclure l'en-tête
    T = ws_test.values.tolist()
    title_T = T[0]
    T = T[1:]  # Correction pour ne pas inclure l'en-tête
    # Nettoyage des données de test (suppression colonnes inutiles)
    for i in T:
        del i[0:1]  # Suppression de la première colonne
    X = np.array(L)
    Y = np.array(T)
    return X , Y
# Eample d'utilisation X,Y = data_processing (ws,ws_test)
