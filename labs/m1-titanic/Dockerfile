# Image de base Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install --no-cache-dir uv

# Copier les fichiers de configuration
COPY pyproject.toml uv.lock README.md ./

# Copier le code source
COPY src/ ./src/

# Installer les dépendances Python
RUN uv sync --frozen --no-dev

# Copier le modèle entraîné
COPY models/titanic_model.pkl ./models/titanic_model.pkl

# Commande par défaut
CMD ["uv", "run", "python", "src/m1_titanic/predict.py"]
