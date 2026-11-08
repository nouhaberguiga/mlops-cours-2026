# MLOps Lab 1 — Titanic Classification

## 1. Présentation du projet

Ce projet consiste à mettre en place un workflow MLOps de bout en bout autour du dataset Titanic.

L'objectif est de construire un modèle de Machine Learning permettant de prédire la survie des passagers du Titanic, tout en assurant le suivi des expériences, la gestion des versions des données et la reproductibilité du pipeline.

Le projet utilise DVC pour le versioning des données et MLflow pour le suivi des expériences et la gestion des modèles.

## 2. Objectifs

- Préparer et nettoyer le dataset Titanic.
- Versionner les données avec DVC.
- Stocker les données dans un stockage objet compatible S3 (SeaweedFS).
- Entraîner plusieurs modèles de classification.
- Réaliser 20 expériences avec différentes configurations.
- Suivre les performances des modèles avec MLflow.
- Enregistrer les modèles dans le Model Registry.
- Reproduire le pipeline avec DVC.

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.11 | Langage de développement |
| uv | Gestion des dépendances et de l'environnement |
| Pandas | Manipulation des données |
| Scikit-learn | Prétraitement et Machine Learning |
| DVC | Versioning des données et pipelines |
| SeaweedFS | Stockage objet compatible S3 |
| MLflow | Suivi des expériences et Model Registry |
| Git / GitHub | Versioning du code source |

## 4. Architecture du projet

```text
m1-titanic/
│
├── data/
│   ├── raw/
│   │   ├── .gitignore
│   │   └── titanic.csv.dvc
│   │
│   └── processed/
│       └── .gitignore
│
├── models/
│   └── .gitignore
│
├── notebooks/
├── reports/
│
├── src/
│   └── m1_titanic/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── predict.py
│       └── grid_search.py
│
├── .dvc/
│   └── config
│
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## 5. Installation

### Prérequis

- Python 3.11
- uv
- Git
- DVC avec le support S3
- Un accès au stockage SeaweedFS
- Un serveur MLflow accessible

### Récupération du projet

Depuis le dépôt :

```bash
cd labs/m1-titanic
```

### Installation des dépendances

```bash
uv sync
```

Cette commande crée l'environnement virtuel et installe les dépendances définies dans `pyproject.toml`.

## 6. Configuration du stockage DVC

Le projet utilise SeaweedFS comme stockage objet compatible S3.

Configuration présente dans `.dvc/config` :

```ini
[core]
    remote = myremote

['remote "myremote"']
    url = s3://dvc-store
    endpointurl = http://localhost:8333
    region = us-east-1
```

Le stockage doit être accessible depuis le poste de travail.

Pour établir le port-forward Kubernetes :

```bash
kubectl -n storage port-forward svc/s3 8333:8333
```

Cette commande doit rester active dans un terminal séparé.

Les identifiants S3 doivent être configurés localement et ne doivent jamais être publiés sur GitHub.

Exemple de configuration locale :

```bash
uv run dvc remote modify --local myremote access_key_id VOTRE_ACCESS_KEY
uv run dvc remote modify --local myremote secret_access_key VOTRE_SECRET_KEY
```

### Récupération des données

Une fois le stockage accessible et les identifiants configurés :

```bash
uv run dvc pull
```

Cette commande récupère les données et les sorties suivies par DVC depuis le stockage distant.

## 7. Prétraitement des données

Le script `src/m1_titanic/preprocessing.py` réalise les opérations suivantes :

- Chargement du dataset Titanic.
- Suppression de la colonne `Cabin`.
- Remplacement des valeurs manquantes de `Age` par la médiane.
- Remplacement des valeurs manquantes de `Embarked` par le mode.
- Sauvegarde du dataset nettoyé dans `data/processed/`.

Exécution :

```bash
uv run python src/m1_titanic/preprocessing.py
```

## 8. Entraînement du modèle

Le script `src/m1_titanic/train.py` entraîne un modèle Random Forest.

Les principales étapes sont :

1. Chargement des données nettoyées.
2. Sélection des variables explicatives.
3. Encodage des variables catégorielles.
4. Séparation des données en ensembles d'entraînement et de test.
5. Entraînement du modèle.
6. Calcul des métriques.
7. Enregistrement des résultats dans MLflow.
8. Sauvegarde du modèle.

Exécution :

```bash
uv run python src/m1_titanic/train.py
```

## 9. Grid Search et suivi des expériences

Le script `src/m1_titanic/grid_search.py` permet de comparer plusieurs configurations de modèles.

Les 20 configurations prévues sont réparties comme suit :

| Algorithme | Nombre de configurations |
|---|---:|
| Random Forest | 6 |
| Decision Tree | 6 |
| Logistic Regression | 3 |
| K-Nearest Neighbors (KNN) | 5 |
| **Total** | **20** |

Les expériences enregistrent notamment les paramètres suivants :

- Modèle utilisé.
- Hyperparamètres.
- Accuracy.
- Precision.
- Recall.
- F1-score.

Le script évite de relancer les configurations déjà terminées avec succès.

Exécution :

```bash
uv run python src/m1_titanic/grid_search.py
```

## 10. MLflow Tracking et Model Registry

Le projet utilise un serveur MLflow accessible à l'adresse :

```text
http://127.0.0.1:5000
```

Pour accéder au serveur MLflow déployé dans Kubernetes :

```bash
kubectl -n mlflow port-forward svc/mlflow 5000:5000
```

Cette commande doit rester active dans un terminal séparé.

Ouvrir ensuite l'interface MLflow :

```text
http://127.0.0.1:5000
```

L'expérience utilisée est :

```text
titanic-classification
```

Le Model Registry contient le modèle :

```text
titanic-classifier
```

Les versions enregistrées permettent de suivre les différentes versions du modèle et leurs étapes de déploiement.

## 11. Pipeline DVC

Le fichier `dvc.yaml` définit deux étapes :

### Étape 1 : preprocess

- Exécution de `preprocessing.py`.
- Lecture du dataset brut.
- Génération du dataset nettoyé.

### Étape 2 : train

- Exécution de `train.py`.
- Entraînement du modèle Random Forest.
- Génération du fichier `models/titanic_model.pkl`.

Représentation du pipeline :

```text
       Dataset Titanic
              |
              v
       Prétraitement
       preprocessing.py
              |
              v
      Dataset nettoyé
              |
              v
        Entraînement
           train.py
              |
              v
       Modèle entraîné
              |
              v
            MLflow
```

### Reproduire le pipeline

```bash
uv run dvc repro
```

Cette commande exécute automatiquement les étapes nécessaires selon les dépendances définies dans `dvc.yaml`.

### Vérifier l'état du pipeline

```bash
uv run dvc status
```

### Envoyer les données vers le stockage distant

```bash
uv run dvc push
```

## 12. Prédiction

Le script `src/m1_titanic/predict.py` est destiné à la prédiction à partir du modèle entraîné.

Pour consulter son fonctionnement et ses arguments :

```bash
uv run python src/m1_titanic/predict.py --help
```

## 13. Commandes principales

| Commande | Description |
|---|---|
| `uv sync` | Installer les dépendances |
| `uv run dvc pull` | Récupérer les données DVC |
| `uv run python src/m1_titanic/preprocessing.py` | Prétraiter les données |
| `uv run python src/m1_titanic/train.py` | Entraîner le modèle |
| `uv run python src/m1_titanic/grid_search.py` | Exécuter les expériences |
| `uv run dvc repro` | Reproduire le pipeline |
| `uv run dvc status` | Vérifier l'état DVC |
| `uv run dvc push` | Envoyer les données vers le stockage distant |

## 14. Conclusion

Ce projet met en pratique les principaux concepts MLOps : versioning des données, automatisation du pipeline, suivi des expériences et gestion des versions des modèles.

L'association de DVC, MLflow, SeaweedFS et Git permet de construire un workflow de Machine Learning plus organisé, traçable et reproductible.
