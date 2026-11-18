# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)

**Binôme :** Travail individuel · **Date :** 12/11/2026 · **Machine :** Dell Latitude 7480, Intel Core i7-7600U, 16 Go RAM, Windows 10 avec Ubuntu 24.04 sous WSL2 ·
**Dataset :** german-credit

## 1. Contexte

Ce laboratoire met en place un pipeline de scoring du risque de crédit composé de six étapes : ingestion des données, validation, préparation des variables, entraînement, évaluation et enregistrement du modèle. Le dataset utilisé est German Credit. Les étapes sont exécutées dans des conteneurs utilisant une image commune, publiée dans le registre local sous le nom scoring-pipeline:1.0.0. Deux orchestrateurs sont étudiés : Kubeflow Pipelines (KFP), déployé sur Kubernetes, et Apache Airflow, exécuté avec Docker Compose. Les deux solutions réutilisent le code des étapes, la configuration du dataset et la même image. Elles diffèrent principalement par la définition, l'exécution et le suivi des workflows.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro (min) | Non mesuré | Non mesuré |
| Temps jusqu'au premier run réussi (min) | Non mesuré | Non atteint |
| Durée d'un run complet (min) | Non mesuré | Non atteint |
| Relance avec les mêmes paramètres (min) | Deux runs réussis, durée non mesurée | Non mesuré |
| Hot-reload : modifier un step et voir le résultat (min) | Non mesuré | Non mesuré |
| Lignes de code propres à l'orchestrateur | Non mesuré | Non mesuré |
| Mémoire consommée par l'outil (Go, docker stats) | Non mesurée | Non mesurée |

Deux exécutions KFP du pipeline ont été confirmées avec le statut SUCCEEDED. Les six étapes ont réussi lors d'une exécution. Les durées précises et la consommation mémoire n'ont toutefois pas été relevées. Le workflow Airflow a été défini, mais son exécution complète n'a pas été validée : une tentative précédente a échoué pendant l'entraînement.

## 3. Analyse qualitative

### Developer experience

KFP propose une définition déclarative du pipeline en composants Python, avec une visualisation claire des dépendances et un suivi des étapes. Il s'intègre naturellement à Kubernetes. Airflow utilise des DAG Python, faciles à comprendre pour les personnes familières avec Python et les workflows de données. Sa prise en main est relativement directe, mais son exécution nécessite une configuration correcte des services et des tâches.

### Scaling

KFP bénéficie des mécanismes de Kubernetes pour distribuer les composants et gérer les ressources des conteneurs. Il convient aux pipelines ML nécessitant plusieurs ressources ou étapes isolées. Airflow peut également paralléliser les tâches grâce à ses exécuteurs, mais le dimensionnement dépend de son architecture et de sa configuration.

### Debugging

KFP permet d'identifier l'étape en échec et de consulter les journaux associés à chaque composant. Airflow fournit une interface pour visualiser les DAG, les dépendances, les états et les logs des tâches. Dans les deux cas, la qualité des logs et la bonne configuration de l'environnement sont essentielles.

## 4. Recommandation

KFP est recommandé pour les projets de Machine Learning qui nécessitent des pipelines reproductibles, des composants conteneurisés, un suivi des expériences et une intégration avec Kubernetes. Airflow est particulièrement adapté à l'orchestration générale des traitements de données, aux dépendances entre tâches et à la planification de workflows variés. Le choix dépend donc du besoin, des compétences de l'équipe et des ressources disponibles. Pour une plateforme ML déjà basée sur Kubernetes, KFP constitue un choix naturel. Pour des workflows de données plus généraux, Airflow est une solution pertinente.

## 5. Limites de la comparaison

Cette comparaison repose sur un laboratoire réalisé sur une seule machine aux ressources limitées. Les temps d'installation, d'exécution, de relance, le hot-reload et la mémoire consommée n'ont pas été mesurés de manière systématique. De plus, le workflow Airflow n'a pas encore atteint une exécution complète réussie. Les résultats ne permettent donc pas d'affirmer qu'un orchestrateur est globalement plus rapide ou plus performant que l'autre. Une comparaison plus fiable nécessiterait plusieurs exécutions contrôlées, des mesures identiques et des tests sur une infrastructure plus représentative d'un environnement de production.
