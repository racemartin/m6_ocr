# ##############################################################################
# BIBLIOTHÈQUES STANDARDS ET TIERS
# ##############################################################################

# Traitement des données
import pandas             as pd                 # Manipulation de DataFrames
import numpy              as np                 # Calculs numériques

# Accès aux bases de données et schémas
from   src.database       import get_engine     # Connexion PostgreSQL
from   src.data.schema    import (              # Définitions du registre
    REGISTRY,
    ColumnType,
    TransformType,
    EncodingType
)

# ##############################################################################
# FONCTIONS GÉNÉRATRICES DE MÉTADONNÉES
# ##############################################################################

def scan_and_generate_specs():
    """
    Scanne la vue SQL pour détecter les colonnes absentes du registre et génère
    le code Python correspondant avec des statistiques de diagnostic.
    """
    # --------------------------------------------------------------------------
    # INITIALISATION ET RÉCUPÉRATION DES DONNÉES
    # --------------------------------------------------------------------------
    
    engine   = get_engine()                     # Moteur de base de données
    self_log = lambda msg: print(f"  → {msg}")  # Utilitaire de log interne
    
    self_log("Lecture de v_features_engineering (échantillon de 5000 lignes)...")
    
    try:
        # Lecture limitée pour optimiser la mémoire lors de l'analyse
        df = pd.read_sql("SELECT * FROM v_features_engineering LIMIT 5000", engine)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de la vue : {e}")
        return

    # --------------------------------------------------------------------------
    # ANALYSE DES DISCORDANCES (REGISTRE VS BASE DE DONNÉES)
    # --------------------------------------------------------------------------

    # Liste des noms techniques déjà enregistrés (en minuscules)
    known_cols = {a.name_technique.lower() for a in REGISTRY.attributes}
    
    # Colonnes à ignorer systématiquement
    excluded   = {'split', 'sk_id_curr', 'index', 'target'}
    
    # Identification des nouvelles colonnes
    unknown    = [c for c in df.columns if c.lower() not in known_cols 
                  and c.lower() not in excluded]
    
    if not unknown:
        self_log("✅ Le registre est synchronisé. Aucune colonne manquante.")
        return

    # --------------------------------------------------------------------------
    # GÉNÉRATION DU RAPPORT ET DU CODE PYTHON
    # --------------------------------------------------------------------------

    print("\n" + "="*80)
    print("GÉNÉRATION AUTOMATIQUE DES SPÉCIFICATIONS D'ATTRIBUTS")
    print("=" * 80)
    
    for col in unknown:
        col_low = col.lower()                   # Normalisation en minuscules
        
        # Statistiques pour le diagnostic de mémoire et qualité
        card     = df[col].nunique()            # Nombre de valeurs uniques
        null_pct = (df[col].isnull().sum() / 
                    len(df)) * 100              # Pourcentage de valeurs nulles
        
        # Détermination de la table source selon le préfixe
        if   col_low.startswith('bureau_'):  src = "v_agg_bureau"
        elif col_low.startswith('prev_'):    src = "v_agg_previous"
        elif col_low.startswith('pos_'):     src = "v_agg_pos_cash"
        elif col_low.startswith('install_'): src = "v_agg_installments"
        elif col_low.startswith('cc_'):      src = "v_agg_credit_card"
        elif col_low.startswith('fe'):       src = "engineered"
        else:                                src = "v_features_engineering"
        
        # ----------------------------------------------------------------------
        # LOGIQUE D'INFERENCE DES TYPES ET ENCODAGES
        # ----------------------------------------------------------------------
        
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            c_type = "ColumnType.CATEGORICAL"
            trans  = "TransformType.NONE"
            # Target Encoding si cardinalité > 10 pour éviter l'explosion RAM
            enc    = f"\n            encoding=EncodingType.{'TARGET' if card > 10 else 'ONE_HOT'},"
        else:
            c_type = "ColumnType.NUMERICAL"
            enc    = ""                         # Pas d'encodage pour le numérique
            # Identification des flux financiers pour transformation logarithmique
            money  = ['amt', 'credit', 'sum', 'debt', 'limit', 'annuity', 'income']
            is_m   = any(x in col_low for x in money)
            trans  = "TransformType.LOG" if is_m else "TransformType.NONE"

        # ----------------------------------------------------------------------
        # AFFICHAGE DU CODE GÉNÉRÉ
        # ----------------------------------------------------------------------
        
        diag   = f" # card={card} | {null_pct:.1f}% null"
        if card > 50: diag += " | ⚠️ HAUTE CARDINALITÉ"

        spec   = (
            f'        build_attribute_spec(\n'
            f'            "{col.upper()}",\n'
            f'            "{col.replace("_", " ").title()}",\n'
            f'            "{col_low}",\n'
            f'            source_table="{src}",\n'
            f'            col_type={c_type},\n'
            f'            transform={trans},{enc}\n'
            f'        ),{diag}'
        )
        print(spec)

if __name__ == "__main__":
    scan_and_generate_specs()