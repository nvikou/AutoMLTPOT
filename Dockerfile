FROM python:3.10-slim

WORKDIR /app

# Dépendances système pour scikit-learn / xgboost
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Par défaut : lancer le script d'export
CMD ["python", "scripts/run_export.py"]
