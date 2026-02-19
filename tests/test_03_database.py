"""
Tests Unitaires: Module Base de données (database.py)
=====================================================

🎯 Niveau: UNIT (+ 1 test INTEGRATION optionnel)
📍 Couche: Infrastructure / Persistance
🏗️  Architecture: Hexagonale - Accès données

Ce module valide:
1. ✅ API du module (get_engine, engine, ensure_postgres_ready, inspect_*)
2. 🔌 Construction de l'URI à partir des variables POSTGRES_*
3. 📋 inspect_table_columns : liste des colonnes, table absente → []
4. 📊 inspect_table_data : DataFrame, limit, table absente → DataFrame vide
5. 🔗 Test d'intégration optionnel : connexion PostgreSQL (skip si indisponible)

MARKERS: @pytest.mark.unit | @pytest.mark.fast | @pytest.mark.integration
"""

# ==============================================================================
# IMPORTS - LIBRAIRIES STANDARD PYTHON
# ==============================================================================
import os                                 # Variables d'environnement, chemins
import sys                                # Accès aux modules chargés (sys.modules)

# ==============================================================================
# IMPORTS - FRAMEWORK DE TESTS ET SIMULATION
# ==============================================================================
import pytest                             # Marqueurs, skip, raises
from   unittest.mock import MagicMock, patch  # Mocks (socket), simulation (env, engine)

# ==============================================================================
# IMPORTS - DONNÉES ET BASE DE DONNÉES
# ==============================================================================
import pandas as pd                       # DataFrame pour les assertions
import sqlalchemy                         # Engine, text() pour SQLite en mémoire

# ##############################################################################
# FIXTURES - CONFIGURATION DES TESTS (SQLite en mémoire, sans PostgreSQL)
# ##############################################################################


def _import_database_with_sqlite():
    """
    Importe src.database avec mocks pour utiliser SQLite en mémoire
    au lieu de PostgreSQL. On patche au niveau sqlalchemy et socket
    pour que les mocks soient actifs dès le chargement du module.
    """
    if "src.database" in sys.modules:
        del sys.modules["src.database"]
    env = {
        "POSTGRES_USER"     : "u",
        "POSTGRES_PASSWORD" : "p",
        "POSTGRES_HOST"     : "localhost",
        "POSTGRES_PORT"     : "5432",
        "POSTGRES_DB"       : "d",
    }
    # Mock socket pour que ensure_postgres_ready ne tente pas de connexion réelle
    mock_sock = MagicMock()
    mock_sock.connect.return_value = None
    mock_socket_cls = MagicMock(return_value=mock_sock)
    mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
    mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
    # create_engine réel pour éviter récursion dans le side_effect
    real_create_engine = sqlalchemy.create_engine
    def _sqlite_engine(*args, **kwargs):
        return real_create_engine("sqlite:///:memory:")
    with patch.dict(os.environ, env, clear=False):
        with patch("socket.socket", mock_socket_cls):
            with patch("sqlalchemy.create_engine", side_effect=_sqlite_engine):
                import src.database as db
                return db


def _make_sqlite_engine():
    """Retourne un engine SQLite en mémoire pour les tests sans Postgres."""
    return sqlalchemy.create_engine("sqlite:///:memory:")


# ##############################################################################
# TEST 01 : API DU MODULE - PRÉSENCE DES FONCTIONS ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_01_module_expose_expected_api():
    """
    Vérifie que le module database expose get_engine, engine et les
    fonctions d'inspection (ensure_postgres_ready, inspect_table_columns,
    inspect_table_data).
    """
    # -------------------------------------------------------------------------
    # ACT: Import du module avec mocks
    # -------------------------------------------------------------------------
    db = _import_database_with_sqlite()

    # -------------------------------------------------------------------------
    # ASSERT: Attributs et callables présents
    # -------------------------------------------------------------------------
    assert hasattr(db, "get_engine")
    assert hasattr(db, "engine")
    assert hasattr(db, "inspect_table_columns")
    assert hasattr(db, "inspect_table_data")
    assert hasattr(db, "ensure_postgres_ready")
    assert callable(db.get_engine)
    assert callable(db.inspect_table_columns)
    assert callable(db.inspect_table_data)

    print("✅ TEST 01 RÉUSSI: API du module database correcte")


# ##############################################################################
# TEST 02 : INSPECT_TABLE_COLUMNS - LISTE DES COLONNES ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_02_inspect_table_columns_returns_column_names():
    """
    inspect_table_columns retourne la liste des noms de colonnes de la table.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Module + table SQLite avec 3 colonnes
    # -------------------------------------------------------------------------
    db     = _import_database_with_sqlite()
    engine = db.engine
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE test_foo (id INTEGER, name TEXT, value REAL)"
            )
        )
        conn.commit()

    # -------------------------------------------------------------------------
    # ACT: Inspection des colonnes
    # -------------------------------------------------------------------------
    columns = db.inspect_table_columns("test_foo", engine)

    # -------------------------------------------------------------------------
    # ASSERT: Liste attendue
    # -------------------------------------------------------------------------
    assert columns == ["id", "name", "value"]

    print("✅ TEST 02 RÉUSSI: Noms de colonnes retournés correctement")


# ##############################################################################
# TEST 03 : INSPECT_TABLE_COLUMNS - NOM NORMALISÉ EN MINUSCULES ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_03_inspect_table_columns_normalizes_table_name_to_lowercase():
    """
    Le nom de table est normalisé en minuscules (table_name.lower()) dans
    la requête ; appel avec TEST_FOO doit trouver la table test_foo.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Table test_foo créée
    # -------------------------------------------------------------------------
    db     = _import_database_with_sqlite()
    engine = db.engine
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text("CREATE TABLE test_foo (id INTEGER)")
        )
        conn.commit()

    # -------------------------------------------------------------------------
    # ACT: Appel avec nom en majuscules
    # -------------------------------------------------------------------------
    columns = db.inspect_table_columns("TEST_FOO", engine)

    # -------------------------------------------------------------------------
    # ASSERT: Table trouvée, une colonne
    # -------------------------------------------------------------------------
    assert "id" in columns
    assert len(columns) == 1

    print("✅ TEST 03 RÉUSSI: Nom de table normalisé en minuscules")


# ##############################################################################
# TEST 04 : INSPECT_TABLE_COLUMNS - TABLE INEXISTANTE → LISTE VIDE ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_04_inspect_table_columns_nonexistent_returns_empty_list():
    """
    Si la table n'existe pas, inspect_table_columns retourne une liste vide.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Module sans table créée
    # -------------------------------------------------------------------------
    db     = _import_database_with_sqlite()
    engine = db.engine

    # -------------------------------------------------------------------------
    # ACT: Table inexistante
    # -------------------------------------------------------------------------
    columns = db.inspect_table_columns("table_inexistante_xyz", engine)

    # -------------------------------------------------------------------------
    # ASSERT: Liste vide
    # -------------------------------------------------------------------------
    assert columns == []

    print("✅ TEST 04 RÉUSSI: Table inexistante → liste vide")


# ##############################################################################
# TEST 05 : INSPECT_TABLE_DATA - RETOURNE UN DATAFRAME ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_05_inspect_table_data_returns_dataframe():
    """
    inspect_table_data retourne un DataFrame avec les lignes de la table.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Table test_bar avec 2 lignes
    # -------------------------------------------------------------------------
    db     = _import_database_with_sqlite()
    engine = db.engine
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE test_bar (id INTEGER, label TEXT)"
            )
        )
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO test_bar (id, label) VALUES (1, 'a'), (2, 'b')"
            )
        )
        conn.commit()

    # -------------------------------------------------------------------------
    # ACT: Lecture des données
    # -------------------------------------------------------------------------
    df = db.inspect_table_data("test_bar", engine, limit=10)

    # -------------------------------------------------------------------------
    # ASSERT: DataFrame avec 2 lignes, colonnes id et label
    # -------------------------------------------------------------------------
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["id", "label"]

    print("✅ TEST 05 RÉUSSI: DataFrame retourné avec les lignes attendues")


# ##############################################################################
# TEST 06 : INSPECT_TABLE_DATA - RESPECT DU PARAMÈTRE LIMIT ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_06_inspect_table_data_respects_limit():
    """
    inspect_table_data limite le nombre de lignes au paramètre limit.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Table avec 10 lignes
    # -------------------------------------------------------------------------
    db     = _import_database_with_sqlite()
    engine = db.engine
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text("CREATE TABLE test_limit (x INTEGER)")
        )
        for i in range(10):
            conn.execute(
                sqlalchemy.text("INSERT INTO test_limit (x) VALUES (:i)"),
                {"i": i},
            )
        conn.commit()

    # -------------------------------------------------------------------------
    # ACT: Limit = 3
    # -------------------------------------------------------------------------
    df = db.inspect_table_data("test_limit", engine, limit=3)

    # -------------------------------------------------------------------------
    # ASSERT: Exactement 3 lignes
    # -------------------------------------------------------------------------
    assert len(df) == 3

    print("✅ TEST 06 RÉUSSI: Paramètre limit respecté")


# ##############################################################################
# TEST 07 : INSPECT_TABLE_DATA - TABLE INEXISTANTE → DATAFRAME VIDE ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_07_inspect_table_data_nonexistent_returns_empty_dataframe():
    """
    Si la table n'existe pas, inspect_table_data retourne un DataFrame vide.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Module sans table
    # -------------------------------------------------------------------------
    db     = _import_database_with_sqlite()
    engine = db.engine

    # -------------------------------------------------------------------------
    # ACT: Table inexistante
    # -------------------------------------------------------------------------
    df = db.inspect_table_data("table_inexistante_abc", engine)

    # -------------------------------------------------------------------------
    # ASSERT: DataFrame vide
    # -------------------------------------------------------------------------
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

    print("✅ TEST 07 RÉUSSI: Table inexistante → DataFrame vide")


# ##############################################################################
# TEST 08 : GET_ENGINE - URI CONSTRUITE À PARTIR DES VARIABLES D'ENVIRONNEMENT ✅
# ##############################################################################


@pytest.mark.unit
@pytest.mark.fast
def test_08_get_engine_builds_uri_from_env():
    """
    get_engine construit l'URI de connexion à partir des variables
    POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB.
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Nettoyage du cache module et env de test
    # -------------------------------------------------------------------------
    if "src.database" in sys.modules:
        del sys.modules["src.database"]
    env = {
        "POSTGRES_USER"     : "myuser",
        "POSTGRES_PASSWORD" : "p@ss",
        "POSTGRES_HOST"     : "dbhost",
        "POSTGRES_PORT"     : "5433",
        "POSTGRES_DB"       : "mydb",
    }

    # -------------------------------------------------------------------------
    # ACT: Import + get_engine (mocks socket + create_engine avant l'import)
    # -------------------------------------------------------------------------
    mock_sock = MagicMock()
    mock_sock.connect.return_value = None
    mock_socket_cls = MagicMock(return_value=mock_sock)
    mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
    mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)
    real_create_engine = sqlalchemy.create_engine
    with patch.dict(os.environ, env, clear=False):
        with patch("socket.socket", mock_socket_cls):
            with patch("sqlalchemy.create_engine") as mock_create:
                mock_create.side_effect = lambda *a, **k: real_create_engine(
                    "sqlite:///:memory:"
                )
                import src.database as db
                db.get_engine()

    # -------------------------------------------------------------------------
    # ASSERT: URI contient les valeurs d'environnement (1er appel = get_engine)
    # -------------------------------------------------------------------------
    uri = mock_create.call_args[0][0]
    assert "postgresql://" in uri
    assert "myuser" in uri
    assert "dbhost" in uri
    assert "5433" in uri
    assert "mydb" in uri

    print("✅ TEST 08 RÉUSSI: URI construite depuis les variables d'environnement")


# ##############################################################################
# TEST 09 : INTEGRATION - CONNEXION POSTGRESQL RÉELLE (OPTIONNEL) 🔗
# ##############################################################################


@pytest.mark.integration
def test_09_postgres_connection_if_available():
    """
    Tente de se connecter à PostgreSQL avec la config du .env.
    Ignoré si le serveur n'est pas disponible (non requis pour la CI).
    """
    # -------------------------------------------------------------------------
    # ARRANGE: Chargement .env
    # -------------------------------------------------------------------------
    from dotenv import load_dotenv
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    if not host or not port:
        pytest.skip("POSTGRES_HOST/POSTGRES_PORT non configurés")

    # -------------------------------------------------------------------------
    # ACT: Connexion réelle (réimport sans mocks)
    # -------------------------------------------------------------------------
    try:
        if "src.database" in sys.modules:
            del sys.modules["src.database"]
        import src.database as db
        with db.engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"PostgreSQL non disponible : {e}")

    # -------------------------------------------------------------------------
    # ASSERT: Engine utilisable
    # -------------------------------------------------------------------------
    assert db.engine is not None

    print("✅ TEST 09 RÉUSSI: Connexion PostgreSQL réelle OK")


# ##############################################################################
# POINT D'ENTRÉE ET RAPPORT
# ##############################################################################


if __name__ == "__main__":
    print("\n============================================================================")
    print("RAPPORT DE TESTS : MODULE BASE DE DONNÉES (database.py)")
    print("============================================================================")
    print("""
    📊 DISTRIBUTION DES TESTS:

    🟢 TESTS UNITAIRES (8 tests - unit + fast):
       ✅ test_01: API du module (get_engine, engine, inspect_*, ensure_postgres_ready)
       ✅ test_02: inspect_table_columns retourne les noms de colonnes
       ✅ test_03: Nom de table normalisé en minuscules
       ✅ test_04: Table inexistante → liste vide
       ✅ test_05: inspect_table_data retourne un DataFrame
       ✅ test_06: Paramètre limit respecté
       ✅ test_07: Table inexistante → DataFrame vide
       ✅ test_08: get_engine construit l'URI depuis POSTGRES_*

    🔗 TEST D'INTÉGRATION (1 test - optionnel):
       ✅ test_09: Connexion PostgreSQL réelle (skip si indisponible)

    📍 COUCHE: Infrastructure / Persistance
    🏗️  ARCHITECTURE: Hexagonale
    """)
    print("============================================================================")

    print("\n🎯 COMMANDES D'EXÉCUTION (uv):")
    print("  Tests unitaires (database).........: uv run python -m pytest tests/test_03_database.py -m unit -v")
    print("  Tests rapides......................: uv run python -m pytest tests/test_03_database.py -m fast -v")
    print("  Sans intégration (CI)...............: uv run python -m pytest tests/test_03_database.py -v -m 'not integration'")
    print("  Avec couverture....................: uv run python -m pytest tests/test_03_database.py -v --cov=src.database")
    print("----------------------------------------------------------------------------\n")
