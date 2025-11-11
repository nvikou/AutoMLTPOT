# AutoMLTPOT

Ce dépôt contient un petit projet d'AutoML basé sur TPOT pour expérimenter la recherche automatique de pipelines de régression. Le workspace est configuré pour être utilisé avec VS Code Dev Containers / GitHub Codespaces et inclut des notebooks Jupyter pour reproduire les étapes de préparation des données, d'entraînement et d'export.
Supposont que nous disposons d'une massive de données contenant une variable d'interet et plusieurs facteurs.
Il est question d entrainner la massive de maniere automatique et de retenir un modele optimal capable de predier la variable d'interet

## Structure du dépôt

- `AutoML_Tpot/` : code source utilitaire (préprocessing, normalisation).
- `notebooks/` : notebooks Jupyter, notamment `ToptAutoML` où sont réalisés les traitements et l'entraînement TPOT.
- `python_boston/` : dossier cible pour l'export du pipeline Python généré par TPOT (`top_bostonkable.py`).
- `data/` : exemples de données (ex : `kable_data.xlsx`).
- `requirements.txt` : dépendances Python nécessaires.
- `scripts/` : dossier qui contient l executable.
- `.devcontainer/` : configuration devcontainer pour VS Code / Codespaces.

## But

Fournir une chaîne reproducible pour :
- expérimenter la recherche automatique,
- charger et préparer les données,
- utiliser TPOT pour trouver un pipeline de régression et de prevision,
- exporter le meilleur pipeline en script Python (dans `python_boston/top_bostonkable.py`).

## Prérequis

- Docker (si vous utilisez Dev Containers localement) ou GitHub Codespaces.
- Python 3.8+ si vous installez localement.

## Installation (locale)

1. Créez et activez un environnement virtuel (recommandé) :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installez les dépendances :

```bash
python3 -m pip install -r requirements.txt
```

Remarque : `requirements.txt` contient notamment `tpot==0.12.2` (version qui expose `score()` et `export()`), `openpyxl` pour la lecture des fichiers Excel, et d'autres bibliothèques usuelles.

## Utilisation avec Dev Container / Codespaces

Le projet inclut `.devcontainer/devcontainer.json`. Lors de la création/reconstruction du conteneur, la commande suivante est exécutée pour installer les dépendances :

```jsonc
"postCreateCommand": "python3 -m pip install -r requirements.txt"
```

Pour appliquer la configuration :

1. Ouvrez la Command Palette (Ctrl+Shift+P) dans VS Code.
2. Choisissez `Dev Containers: Rebuild and Reopen in Container`.

Cela créera un environnement contenant toutes les dépendances listées.

## Exécution rapide (notebook)

1. Ouvrez `notebooks/ToptAutoML.ipynb`.
2. Exécutez la première cellule qui ajoute la racine du projet au `sys.path` (ou assurez-vous d'ouvrir le notebook depuis la racine du repo). Cela permet d'importer `AutoML_Tpot` sans installer le package.
3. Exécutez la cellule de contrôle (chargement des données) pour vérifier que `data/kable_data.xlsx` est lisible et que les colonnes sont correctes.
4. Pour un test rapide d'entraînement (validation de la chaîne) : utilisez `population_size=10, generations=1`.

Exemple de cellule (déjà présente dans le notebook) :

```python
reg = TPOTRegressor(verbosity=2, population_size=10, generations=1, random_state=35)
reg.fit(X_train, y_train)
print('score:', reg.score(X_test, y_test))
reg.export('python_boston/top_bostonkable.py')
```

Après exécution, le fichier `python_boston/top_bostonkable.py` doit être créé (si l'export réussit).

## A noter que generations=1 a ete utilisé a titre experimental. On pourra utiliser generations= 50, generations=100...

## Export du pipeline

- TPOT 0.12.2 fournit la méthode `export(<path>)` qui génère un script Python contenant le pipeline. Vérifiez le fichier généré et corrigez les imports si nécessaire (ex : changement de chemin, dépendances non-importées automatiquement).
- Si `export()` échoue, la stratégie de secours est de sauvegarder l'objet `reg.fitted_pipeline_` ou l'objet `reg` avec `joblib.dump()` pour réutilisation ultérieure.

## Résolution des problèmes courants

- Import local non trouvé dans le notebook : ajoutez au début du notebook :

```python
import sys, os
sys.path.insert(0, os.getcwd())  # ou le chemin absolu vers la racine du repo
```

- `pd.read_excel` échoue : installez `openpyxl` (déjà présent dans `requirements.txt`).
- TPOT renvoie une erreur `train set will be empty` : vérifiez la variable cible (`y`) après preprocessing — il peut y avoir beaucoup de NaN. Inspectez `y` avec `y.isna().sum()` et la distribution des valeurs.

## Exemples de commandes utiles

```bash
# installer les dépendances
python3 -m pip install -r requirements.txt

# reconstruire le devcontainer (dans VS Code)
# ouvrez la palette → Dev Containers: Rebuild and Reopen in Container

# exécuter le notebook localement (optionnel)
jupyter lab notebooks/ToptAutoML.ipynb
```


## Licence

