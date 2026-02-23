from sqlalchemy    import text
from pathlib       import Path
from src.database  import get_engine

CURRENT_DIR = Path(__file__).parent
SQL_PATH    = CURRENT_DIR / "create_db_infrastructure_schema.sql"

def run_infrastructure_setup():
    """
    Initialise les tables de l'infrastructure en utilisant le moteur SQL global.
    """
    engine = get_engine()

    try:
        # 1. Lecture du fichier SQL
        if not SQL_PATH.exists():
            raise FileNotFoundError(f"Le fichier {SQL_PATH} est introuvable.")

        with open(SQL_PATH, 'r', encoding='utf-8') as file:
            # Filtre pour ignorer les commandes spécifiques à la console psql (\c)
            sql_content = "\n".join(line for line in file if not line.strip().startswith("\\c"))

        # 2. Exécution et Vérification
        print(f"🚀 Déploiement du schéma technique vers la base : {engine.url.database}...")

        with engine.begin() as conn:
            # A. Exécution du script de création
            conn.execute(text(sql_content))

            # B. Vérification immédiate (La "touche Maestro")
            # On interroge le catalogue système de PostgreSQL pour lister les tables créées
            query = text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';")
            result = conn.execute(query)
            tables = [row[0] for row in result]

            print(f"📊 Tables détectées dans la DB : {', '.join(tables)}")

        print("✅ Succès : L'infrastructure est prête et vérifiée.")

    except Exception as e:
        print(f"❌ Erreur critique lors de l'initialisation : {e}")

if __name__ == "__main__":
    run_infrastructure_setup()
