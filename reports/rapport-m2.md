# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)

**Binôme :** Travail individuel  
**Date :** 21/11/2026  
**Machine :** Dell Latitude 7480, Intel Core i7-7600U, 16 Go RAM, Windows avec Ubuntu 24.04 sous WSL2  
**Dataset :** German Credit

## 1. Contexte

Ce laboratoire consiste à comparer deux orchestrateurs pour un pipeline de scoring du risque de crédit : Kubeflow Pipelines (KFP) et Apache Airflow. Le workflow comprend six étapes : ingestion des données, validation, préparation des variables, entraînement, évaluation et enregistrement du modèle.

Les deux solutions utilisent le même dataset, la même image Docker et le code commun des étapes. KFP est déployé sur Kubernetes, tandis qu'Airflow fonctionne avec Docker Compose. L'objectif est de comparer leur expérience de développement, leur exécution et leur facilité de supervision.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro (min) | Non mesuré | Non mesuré |
| Temps jusqu'au premier run réussi (min) | Non mesuré | Non mesuré |
| Durée d'un run complet (min) | Non mesurée | Non mesurée de façon fiable |
| Relance avec les mêmes paramètres | Deux runs réussis, durée non mesurée | Non mesurée |
| Hot-reload (min) | Non mesuré | Non mesuré |
| Lignes dans les fichiers d'orchestration | 228 | 77 |
| Mémoire observée (`docker stats`) | Non isolable | Environ 1,1 Gio |

Les deux exécutions KFP ont été confirmées avec le statut SUCCEEDED. L'intégration MLflow a également été vérifiée : un run KFP réussi est associé à une expérience MLflow et une version du modèle scoring-model a été produite.

Du côté Airflow, le DAG scoring_dag a terminé avec succès. Les six tâches du workflow ont été exécutées avec succès et un run MLflow portant le tag launched_by=airflow a été confirmé. Les tests locaux de téléchargement et de validation ont également réussi : 1 000 lignes ont été traitées, avec six contrôles réussis et aucune erreur.

Les 228 lignes correspondent aux fichiers pipelines/components.py et pipelines/scoring_pipeline.py ; les 77 lignes correspondent à dags/scoring_dag.py. Ces valeurs représentent la taille des fichiers, et non uniquement le code spécifique à chaque orchestrateur. La mémoire Airflow est une estimation incluant ses services Docker, notamment PostgreSQL et Redis. La mémoire KFP n'a pas pu être isolée, car ses composants partagent les conteneurs Kind avec d'autres services Kubernetes.

## 3. Analyse qualitative

### Developer experience

KFP permet de définir le workflow sous forme de composants Python et de visualiser les dépendances entre les étapes. Son intégration avec Kubernetes facilite l'exécution de composants conteneurisés. Airflow utilise des DAG Python, dont la structure est relativement simple à comprendre. Son interface permet de suivre les tâches et leurs journaux.

### Scaling

KFP bénéficie des mécanismes de Kubernetes pour gérer les ressources et distribuer les composants. Il convient aux pipelines ML conteneurisés. Airflow permet également la parallélisation des tâches, selon l'exécuteur et les ressources configurées.

### Debugging

KFP fournit des informations sur l'état des composants et leurs logs. Airflow permet de visualiser les DAG, les dépendances, les états et les logs de chaque tâche. Dans les deux solutions, la configuration des environnements et la qualité des logs sont importantes pour identifier les erreurs.

## 4. Recommandation

KFP est recommandé pour les projets de Machine Learning nécessitant des composants conteneurisés, une intégration Kubernetes et un suivi des expériences. Airflow est adapté à l'orchestration générale des traitements de données et aux workflows comportant des dépendances variées. Le choix dépend des besoins du projet, des compétences de l'équipe et de l'infrastructure disponible.

Pour une plateforme ML déjà basée sur Kubernetes, KFP constitue un choix naturel. Pour des traitements de données plus généraux, Airflow représente une solution pertinente.

## 5. Limites de la comparaison

Cette comparaison a été réalisée sur une seule machine disposant de ressources limitées. Les temps d'installation, d'exécution, de relance et de hot-reload n'ont pas été mesurés dans des conditions identiques. La mémoire des composants KFP n'a pas pu être séparée de celle des autres services Kubernetes. Les mesures de mémoire ne sont donc pas directement comparables.

Une étude plus précise nécessiterait plusieurs exécutions contrôlées, des mesures de durée et de consommation réalisées avec la même méthode, ainsi que des tests sur une infrastructure plus représentative de la production.
