"""Script d'extraction et d'export TPOT

- charge les deux feuilles Excel attendues
- appelle AutoML_Tpot.data_processing.data_processing
- affiche des diagnostics sur la variable cible
- si la cible contient des valeurs numériques valides, lance un TPOT rapide
- exporte le pipeline dans python_boston/top_bostonkable.py (ou sauvegarde joblib en secours)

Usage:
    python3 scripts/run_export.py
"""

import os
import sys
import traceback
from pathlib import Path

# s'assurer que la racine du repo est dans sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

# imports TPOT et utilitaires
try:
    from tpot import TPOTRegressor
except Exception:
    TPOTRegressor = None

import joblib

# import user preprocessing
try:
    from AutoML_Tpot import data_processing
except Exception:
    data_processing = None


def load_sheets(path):
    xlsx = pd.ExcelFile(path)
    # essayer de détecter les deux feuilles utilisées précédemment
    sheets = xlsx.sheet_names
    # préférer les noms connus
    sheet1 = 'Massive for learning' if 'Massive for learning' in sheets else sheets[0]
    sheet2 = 'Massive' if 'Massive' in sheets else (sheets[1] if len(sheets) > 1 else sheets[0])
    df1 = pd.read_excel(path, sheet_name=sheet1)
    df2 = pd.read_excel(path, sheet_name=sheet2)
    return df1, df2, sheet1, sheet2


def main():
    try:
        data_path = ROOT / 'data' / 'kable_data.xlsx'
        if not data_path.exists():
            print('Fichier attendu non trouvé:', data_path)
            return 3

        print('Chargement des feuilles Excel...')
        ws, ws_test, s1, s2 = load_sheets(data_path)
        print('Feuilles:', s1, '=>', ws.shape, ',', s2, '=>', ws_test.shape)

        if data_processing is None:
            print("Impossible d'importer AutoML_Tpot.data_processing. Vérifiez sys.path ou installez le package.")
            return 4

        print("Appel de data_processing(ws, ws_test) ...")
        try:
            result = data_processing.data_processing(ws, ws_test)
        except Exception as e:
            print("Erreur lors de l'appel à data_processing:", e)
            traceback.print_exc()
            return 5

        # data_processing peut renvoyer X, Y ou autre chose; essayer de déballer
        if isinstance(result, tuple) and len(result) >= 1:
            X = result[0]
            Y = result[1] if len(result) > 1 else None
        else:
            print("data_processing n'a pas renvoyé un tuple attendu. Valeur renvoyée:", type(result))
            return 6

        X = np.asarray(X)
        print('X shape après data_processing:', X.shape)

        # Par convention existante dans notebook: y_target = X[:,0]
        if X.shape[1] < 1:
            print("X n'a pas de colonne 0 pour la cible.")
            return 7

        y_raw = X[:, 0]
        print('Exemples bruts de y (10 premiers):', y_raw[:10])

        # convertir en numérique
        y_numeric = pd.to_numeric(pd.Series(y_raw), errors='coerce')
        n_total = len(y_numeric)
        n_nonnull = y_numeric.notna().sum()
        n_null = n_total - n_nonnull
        print(f'y total={n_total}, non-nulls={n_nonnull}, nulls={n_null}')

        if n_nonnull == 0:
            print('Aucune valeur numérique valide pour y — arrêt. Inspectez data_processing ou la sélection de la cible.')
            # sauvegarder un échantillon pour debug
            debug_dir = ROOT / 'debug'
            debug_dir.mkdir(exist_ok=True)
            pd.DataFrame(X[:20, :]).to_csv(debug_dir / 'X_sample_head.csv', index=False)
            ws.head(20).to_csv(debug_dir / 'ws_head.csv', index=False)
            ws_test.head(20).to_csv(debug_dir / 'ws_test_head.csv', index=False)
            print('Échantillons CSV sauvegardés dans:', debug_dir)
            return 8

        # filtrer les lignes valides
        valid_idx = y_numeric.notna().values
        X_valid = X[valid_idx]
        y_valid = y_numeric[valid_idx].values
        print('Après filtrage: X_valid.shape =', X_valid.shape, 'y_valid.shape =', y_valid.shape)

        # préparer X_features suivant la convention existante (X[:,2:]) si possible
        if X_valid.shape[1] >= 3:
            X_features = X_valid[:, 2:]
        else:
            # si peu de colonnes, prendre toutes sauf la cible
            X_features = X_valid[:, 1:]
        print('X_features.shape =', X_features.shape)

        # conversion en float
        X_features = np.asarray(pd.DataFrame(X_features).apply(pd.to_numeric, errors='coerce')).astype(float)

        # split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_valid, test_size=0.25, random_state=42)
        print('Split fait: train=', X_train.shape, 'test=', X_test.shape)

        # vérifier TPOT disponible
        if TPOTRegressor is None:
            print('TPOT non importable. Installez tpot==0.12.2 ou équivalent.')
            return 9

        # entraînement rapide
        print('Lancement TPOT (rapide) ...')
        reg = TPOTRegressor(generations=1, population_size=10, verbosity=2, random_state=35, n_jobs=1)
        reg.fit(X_train, y_train)

        # score si disponible
        try:
            score = reg.score(X_test, y_test)
            print('Score (r2) sur test set:', score)
        except Exception:
            print('reg.score() indisponible ou erreur lors du calcul du score.')

        # export
        out_dir = ROOT / 'python_boston'
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / 'top_bostonkable.py'
        try:
            reg.export(str(out_path))
            print('Export TPOT effectué:', out_path)
        except Exception:
            print('reg.export a échoué, sauvegarde joblib en secours.')
            joblib.dump(reg.fitted_pipeline_ if hasattr(reg, 'fitted_pipeline_') else reg, out_dir / 'top_bostonkable.joblib')
            print('Sauvegarde joblib faite dans', out_dir)

        return 0

    except Exception as e:
        print('Erreur inattendue:', e)
        traceback.print_exc()
        return 10


if __name__ == '__main__':
    sys.exit(main())
