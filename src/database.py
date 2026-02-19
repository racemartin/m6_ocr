import os
import socket
import subprocess
import time
import urllib.parse
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def ensure_postgres_ready(host, port):
    """Vérifie si Postgres accepte les connexions ; sinon, tente de le démarrer sous Windows."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        try:
            s.connect((host, int(port)))
            return True
        except (ConnectionRefusedError, TimeoutError, socket.timeout):
            print(f"⚠️ PostgreSQL sur {host}:{port} ne répond pas. Tentative de démarrage du service...")
            try:
                # Tente de démarrer le service sous Windows
                subprocess.run(
                    ["powershell", "Start-Service postgresql*"],
                    check=True,
                    capture_output=True,
                )
                print("🚀 Service PostgreSQL démarré. En attente de stabilité...")
                time.sleep(5)  # Pause pour que le processus ouvre le socket
                return True
            except Exception as e:
                print(f"❌ Erreur lors du démarrage de PostgreSQL : {e}")
                return False


def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = urllib.parse.quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    dbname = os.getenv("POSTGRES_DB")

    # Vérification de la disponibilité avant de créer l'engine
    ensure_postgres_ready(host, port)

    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(db_uri, client_encoding="utf8")


# Instance globale
engine = get_engine()


def inspect_table_columns(table_name, engine):
    """
    Retourne la liste des noms de colonnes d'une table donnée.
    Compatible PostgreSQL (information_schema) et SQLite (PRAGMA table_info).
    """
    table = table_name.lower()
    try:
        if engine.dialect.name == "sqlite":
            # SQLite : PRAGMA table_info retourne cid, name, type, notnull, ...
            query = f"PRAGMA table_info({table!r});"
            col_name_key = "name"
        else:
            # PostgreSQL (et autres) : information_schema.columns
            query = (
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{table}';"
            )
            col_name_key = "column_name"

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            print(f"⚠️ La table '{table_name}' n'existe pas.")
            return []  # Retourne toujours une liste, même vide

        return df[col_name_key].tolist()
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return []


def inspect_table_data(table_name, engine, limit=5):
    """
    Retourne les premiers enregistrements d'une table ou vue PostgreSQL.
    Équivalent de df.head() pour la base de données.
    """
    try:
        query = f"SELECT * FROM {table_name.lower()} LIMIT {limit};"
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df  # Pandas renvoie un DF vide s'il n'y a pas de données
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de la table '{table_name}' : {e}")
        return pd.DataFrame()  # Retourne toujours un DataFrame
