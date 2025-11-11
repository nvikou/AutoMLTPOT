import pandas as pd
import numpy as np


# Fonction de normalisation des prédictions par mois
# Elle me permet de m'assurer que la somme des prédictions pour chaque mois complet est égale à 1.
# Les mois incomplets conservent leurs prédictions d'origine.

def normalize_predictions_by_month(ws_test, predictions):
    """Normalise les prédictions pour que la somme par mois soit égale à 1,
       uniquement pour les mois complets sont pris en compte. Les mois incomplets conservent les prédictions d'origine.
       Elle me permet de m'assurer que la somme des prédictions pour chaque mois complet est égale à 1."""
    df_test_original = ws_test.copy()
    df_test_original['Period'] = pd.to_datetime(df_test_original['Period'], errors='coerce')

    min_len = min(len(df_test_original), len(predictions))
    df_test_original = df_test_original.iloc[:min_len].copy()

    # Ajouter les prédictions brutes
    df_test_original['Prédiction_raw'] = predictions[:min_len]

    # Extraire la période mensuelle
    df_test_original['Mois'] = df_test_original['Period'].dt.to_period('M')

    # Identifier les mois complets : nombre de lignes = nombre de jours dans le mois
    counts_per_month = df_test_original.groupby('Mois').size()
    days_in_month = df_test_original['Period'].dt.days_in_month.groupby(df_test_original['Mois']).first()
    complete_months = counts_per_month[counts_per_month == days_in_month].index

    # Créer une colonne Prédiction initialisée à Prédiction_raw
    df_test_original['Prédiction'] = df_test_original['Prédiction_raw']

    # Appliquer la normalisation uniquement aux lignes des mois complets
    mask_complete = df_test_original['Mois'].isin(complete_months)
    df_complete = df_test_original[mask_complete].copy()

    if not df_complete.empty:
        monthly_sum = df_complete.groupby('Mois')['Prédiction_raw'].transform('sum')
        monthly_sum = monthly_sum.replace(0, 1)  # éviter division par zéro

        df_test_original.loc[mask_complete, 'Prédiction'] = df_complete['Prédiction_raw'] / monthly_sum

        # Affichage vérification
        print("Сумма прогнозов по месяцам  (полные месяцы) :")
        print(df_test_original[mask_complete].groupby('Mois')['Prédiction'].sum())
    else:
        print("Aucun mois complet trouvé pour la normalisation.")

    # Les mois incomplets conservent leurs Prédiction_raw non normalisées
    incomplete_months = counts_per_month[counts_per_month != days_in_month].index
    if len(incomplete_months) > 0:
        print("Неполные месяцы :")
        for m in incomplete_months:
            print(f"  - {m} (количество дней : {counts_per_month[m]}, ожидаемые дни : {days_in_month[m]})")

    return df_test_original

    
# Exemple d'utilisation : normalized = normalize_predictions_by_month(ws_test, predictions)