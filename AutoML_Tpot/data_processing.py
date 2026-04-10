import pandas as pd
import numpy as np


def data_processing(ws, ws_test):
    """Prépare les données d'entraînement et de test.

    pd.read_excel() place déjà la première ligne en en-têtes,
    donc ws.values ne contient que les données.

    Retourne X (entraînement) et Y (test) sous forme de tableaux numpy.
    La première colonne (souvent un identifiant / date) est supprimée
    des DEUX jeux pour garder la cohérence.
    """
    X = ws.values  # pas de skip de la 1ère ligne — pandas l'a déjà fait
    Y = ws_test.values

    # Suppression de la première colonne (identifiant) sur les deux jeux
    X = X[:, 1:]
    Y = Y[:, 1:]

    return np.array(X), np.array(Y)
