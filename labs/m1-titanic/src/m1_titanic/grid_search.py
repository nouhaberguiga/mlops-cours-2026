import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_data():
    df = pd.read_csv("data/processed/titanic_clean.csv")

    features = [
        "Pclass", "Sex", "Age",
        "SibSp", "Parch", "Fare", "Embarked"
    ]

    X = df[features]
    y = df["Survived"]

    X = pd.get_dummies(
        X,
        columns=["Sex", "Embarked"],
        drop_first=True
    )

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


def run_experiment(model, model_name, params, X_train, X_test, y_train, y_test):

    with mlflow.start_run():

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_param("model", model_name)
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.metrics._dist_metrics.EuclideanDistance64",
                "sklearn.neighbors._kd_tree.KDTree"
            ]
        )

        print(
            f"{model_name} | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1-score: {f1:.4f}"
        )


def main():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("titanic-classification")

    X_train, X_test, y_train, y_test = load_data()

    experiments = []

    # 1. Random Forest : 6 expériences
    for n in [50, 100, 150]:
        for depth in [5, 10]:
            params = {
                "n_estimators": n,
                "max_depth": depth
            }

            model = RandomForestClassifier(
                **params,
                random_state=42
            )

            experiments.append((model, "RandomForest", params))

    # 2. Decision Tree : 6 expériences
    for depth in [3, 5, 10]:
        for criterion in ["gini", "entropy"]:
            params = {
                "max_depth": depth,
                "criterion": criterion
            }

            model = DecisionTreeClassifier(
                **params,
                random_state=42
            )

            experiments.append((model, "DecisionTree", params))

    # 3. Logistic Regression : 3 expériences
    for c in [0.1, 1, 10]:
        params = {"C": c}

        model = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=42
        )

        experiments.append((model, "LogisticRegression", params))

    # 4. KNN : 5 expériences
    for k in [3, 5, 7, 9, 11]:
        params = {"n_neighbors": k}

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        experiments.append((model, "KNN", params))

    print(f"\nNombre total d'expériences prévues : {len(experiments)}")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name("titanic-classification")

    existing_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000
    )

    completed_configs = set()

    for run in existing_runs:
        if run.info.status == "FINISHED":
            model_name = run.data.params.get("model")

            params = {
                key: value
                for key, value in run.data.params.items()
                if key != "model" and key != "random_state"
            }

            completed_configs.add(
                (model_name, tuple(sorted(params.items())))
            )

    print(f"Configurations déjà réussies : {len(completed_configs)}")

    success_count = 0

    for model, model_name, params in experiments:

        config_key = (
            model_name,
            tuple(sorted(
                (key, str(value))
                for key, value in params.items()
            ))
        )

        if config_key in completed_configs:
            print(f"Déjà exécutée : {model_name} {params}")
            continue

        print(f"\nExécution : {model_name} {params}")

        try:
            run_experiment(
                model,
                model_name,
                params,
                X_train,
                X_test,
                y_train,
                y_test
            )

            success_count += 1

        except Exception as e:
            print(f"Erreur pour {model_name} avec {params} : {e}")

    print(f"\nNouvelles expériences réussies : {success_count}")
    print("Grid Search terminé !")

if __name__ == "__main__":
    main()
