import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import joblib


def train_model():

    # Configuration MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("titanic-classification")

    with mlflow.start_run():

        # Charger les données
        df = pd.read_csv("data/processed/titanic_clean.csv")

        # Sélectionner les variables
        features = [
            "Pclass", "Sex", "Age",
            "SibSp", "Parch", "Fare", "Embarked"
        ]

        X = df[features]
        y = df["Survived"]

        # Encodage des variables catégorielles
        X = pd.get_dummies(
            X, columns=["Sex", "Embarked"], drop_first=True
        )

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Paramètres du modèle
        n_estimators = 100
        random_state = 42

        # Entraînement
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # Prédictions
        predictions = model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Affichage
        print("Accuracy :", accuracy)
        print("\nClassification Report :")
        print(classification_report(y_test, predictions))

        # Enregistrement des paramètres dans MLflow
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Enregistrement des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Enregistrement du modèle dans MLflow
        mlflow.sklearn.log_model(
   	       	model,
   		 name="model",
    		skops_trusted_types=["sklearn.tree._tree.Tree"]
	)

        # Sauvegarde locale
        joblib.dump(model, "models/titanic_model.pkl")

        print("\nModèle sauvegardé avec succès !")
        print("Expérience enregistrée dans MLflow.")


if __name__ == "__main__":
    train_model()
