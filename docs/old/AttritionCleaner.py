import pandas as pd
from sqlalchemy import text
from notebooks.DataCleaner import DataCleaner

class AttritionCleaner(DataCleaner):


    def __init__(self, engine):
        super().__init__(df=None, verbose=True)
        self.engine = engine

    # --- MÉTODOS AUXILIARES GENÉRICOS ---
    def _execute(self, query, message):
        with self.engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()
        print(f"✅ {message}")

    # ##########################################################################
    def ____BLOC_SIRH(self): pass

    # ##########################################################################
    # BLOC 1 : GESTION DES DONNÉES SIRH (SYSTÈME D'INFORMATION RH)
    # ##########################################################################
    # Structure originale du CSV (raw_sirh) :
    # 0. id_employee                    : int64 (Clé technique)
    # 1. age                            : int64 (Âge actuel)
    # 2. genre                          : object (Sexe de l'employé)
    # 3. revenu_mensuel                 : int64 (Salaire brut)
    # 4. statut_marital                 : object (Situation familiale)
    # 5. departement                    : object (Direction d'affectation)
    # 6. poste                          : object (Intitulé du poste)
    # 7. nombre_experiences_precedentes : int64 (Nb entreprises passées)
    # 8. nombre_heures_travailless      : int64 (Base contrat)
    # 9. annee_experience_totale        : int64 (Cumul carrière)
    # 10. annees_dans_l_entreprise      : int64 (Ancienneté boîte)
    # 11. annees_dans_le_poste_actuel   : int64 (Ancienneté poste)
    # 12. emp_id                        : object (Clé métier générée)
    # ##########################################################################

    def create_raw_sirh(self):
        """Crée la table physique pour l'ingestion brute du SIRH."""
        query = """
        DROP TABLE IF EXISTS raw_sirh CASCADE;
        CREATE TABLE IF NOT EXISTS raw_sirh (
            id_employee                    INT PRIMARY KEY,
            age                            INT,
            genre                          VARCHAR(10),
            revenu_mensuel                 INT,
            statut_marital                 VARCHAR(30),
            departement                    VARCHAR(50),
            poste                          VARCHAR(100),
            nombre_experiences_precedentes INT,
            nombre_heures_travailless      INT,
            annee_experience_totale        INT,
            annees_dans_l_entreprise        INT,
            annees_dans_le_poste_actuel    INT,
            emp_id                         VARCHAR(10)
        );"""
        self._execute(query, "Table 'raw_sirh' créée ou déjà existante.")

    def clean_sirh(self):
        """Nettoie les espaces et types pour les données SIRH."""
        sql = """
        DROP VIEW IF EXISTS v_clean_sirh CASCADE;
        CREATE VIEW v_clean_sirh AS
        SELECT 
            CAST(emp_id AS VARCHAR(10)) AS emp_id,
            id_employee,
            age,
            TRIM(genre)                 AS genre,
            revenu_mensuel,
            TRIM(statut_marital)        AS statut_marital,
            TRIM(departement)           AS departement,
            TRIM(poste)                 AS poste,
            nombre_experiences_precedentes,
            nombre_heures_travailless,
            annee_experience_totale,
            annees_dans_l_entreprise,
            annees_dans_le_poste_actuel
        FROM raw_sirh WHERE emp_id IS NOT NULL;
        """
        self._execute(sql, "Vue 'v_clean_sirh' générée avec succès.")

    # ##########################################################################
    def ____BLOC_EVAL(self): pass

   # ##########################################################################
    # BLOC 2 : GESTION DES ÉVALUATIONS (EVALS)
    # ##########################################################################
    # Structure originale du CSV (raw_evals) :
    # 0. satisfaction_employee_environnement       : int64 (Note 1-4)
    # 1. note_evaluation_precedente                : int64 (Note performance)
    # 2. niveau_hierarchique_poste                 : int64 (Rang manager)
    # 3. satisfaction_employee_nature_travail      : int64 (Motivation)
    # 4. satisfaction_employee_equipe              : int64 (Climat social)
    # 5. satisfaction_employee_equilibre_pro_perso : int64 (WLB)
    # 6. eval_number                               : object (Lien ID)
    # 7. note_evaluation_actuelle                  : int64 (Performance N)
    # 8. heure_supplementaires                     : object (Oui/Non)
    # 9. augementation_salaire_precedente          : object (Pourcentage %)
    # ##########################################################################

    def create_raw_evals(self):
        """Définit le schéma de réception pour les évaluations annuelles."""
        query = """
        DROP TABLE IF EXISTS raw_evals CASCADE;
        CREATE TABLE IF NOT EXISTS raw_evals (
            satisfaction_employee_environnement       INT,
            note_evaluation_precedente                INT,
            niveau_hierarchique_poste                 INT,
            satisfaction_employee_nature_travail      INT,
            satisfaction_employee_equipe              INT,
            satisfaction_employee_equilibre_pro_perso INT,
            eval_number                               VARCHAR(20),
            note_evaluation_actuelle                  INT,
            heure_supplementaires                     VARCHAR(10),
            augementation_salaire_precedente          VARCHAR(10),
            emp_id                                    VARCHAR(10)
        );"""
        self._execute(query, "Table 'raw_evals' initialisée.")

    def clean_evals(self):
        """Transforme les variables catégorielles et scalaires des évaluations."""
        sql = """
        DROP VIEW IF EXISTS v_clean_evals CASCADE;
        CREATE VIEW v_clean_evals AS
        SELECT 
            CAST(emp_id AS VARCHAR(10)) AS emp_id,
            satisfaction_employee_environnement,
            note_evaluation_precedente,
            niveau_hierarchique_poste,
            satisfaction_employee_nature_travail,
            satisfaction_employee_equipe,
            satisfaction_employee_equilibre_pro_perso,
            note_evaluation_actuelle,
            CASE 
                WHEN TRIM(heure_supplementaires) IN ('Oui', 'Yes') THEN 1 
                ELSE 0 
            END AS heure_supplementaires,
            CASE 
                WHEN augementation_salaire_precedente ~ '^[0-9]' THEN 
                    CAST(REPLACE(
                        augementation_salaire_precedente, '%', ''
                    ) AS FLOAT) / 100
                ELSE 0 
            END AS augementation_salaire_precedente
        FROM raw_evals;
        """
        self._execute(sql, "Vue 'v_clean_evals' (Preprocessing SQL) créée.")

    # ##########################################################################
    def ____BLOC_SONDAGE(self): pass

    # ##########################################################################
    # BLOC 3 : GESTION DES SONDAGES (SONDAGE)
    # ##########################################################################
    # Structure originale du CSV (raw_sondage) :
    # 0. a_quitte_l_entreprise              : object (Cible : Target)
    # 1. nombre_participation_pee            : int64 (Plans épargne)
    # 2. nb_formations_suivies               : int64 (Formations N-1)
    # 3. nombre_employee_sous_responsabilite : int64 (Taille équipe gérée)
    # 4. code_sondage                        : int64 (Lien ID)
    # 5. distance_domicile_travail           : int64 (Kilomètres)
    # 6. niveau_education                    : int64 (Diplôme 1-5)
    # 7. domaine_etude                       : object (Filière)
    # 8. ayant_enfants                       : object (Charge famille)
    # 9. frequence_deplacement               : object (Voyages pro)
    # 10. annees_depuis_la_derniere_promotion: int64 (Stagnation)
    # 11. annes_sous_responsable_actuel      : int64 (Stabilité management)
    # ##########################################################################

    def create_raw_sondage(self):
        """Crée la table pour les données issues des enquêtes d'engagement."""
        query = """
        DROP TABLE IF EXISTS raw_sondage CASCADE;
        CREATE TABLE IF NOT EXISTS raw_sondage (
            a_quitte_l_entreprise               VARCHAR(10),
            nombre_participation_pee             INT,
            nb_formations_suivies                INT,
            nombre_employee_sous_responsabilite  INT,
            code_sondage                         INT,
            distance_domicile_travail            INT,
            niveau_education                     INT,
            domaine_etude                        VARCHAR(100),
            ayant_enfants                        VARCHAR(10),
            frequence_deplacement                VARCHAR(50),
            annees_depuis_la_derniere_promotion  INT,
            annes_sous_responsable_actuel        INT,
            emp_id                               VARCHAR(10)
        );"""
        self._execute(query, "Table 'raw_sondage' prête pour ingestion.")

    def clean_sondage(self):
        """Nettoie et normalise les variables cibles et catégorielles du sondage."""
        sql = """
        DROP VIEW IF EXISTS v_clean_sondage CASCADE;
        CREATE VIEW v_clean_sondage AS
        SELECT 
            CAST(emp_id AS VARCHAR(10)) AS emp_id,
            CASE 
                WHEN TRIM(a_quitte_l_entreprise) IN ('Oui', 'Yes') THEN 1
                ELSE 0 
            END AS target_attrition,
            nombre_participation_pee,
            nb_formations_suivies,
            nombre_employee_sous_responsabilite,
            distance_domicile_travail,
            niveau_education            AS niveau_education,
            TRIM(domaine_etude)         AS domaine_etude,
            CASE 
                WHEN TRIM(ayant_enfants) IN ('Oui', 'Yes') THEN 1 
                ELSE 0 
            END AS ayant_enfants,
            TRIM(frequence_deplacement) AS frequence_deplacement,
            annees_depuis_la_derniere_promotion,
            annes_sous_responsable_actuel
        FROM raw_sondage WHERE emp_id IS NOT NULL;
        """
        self._execute(sql, "Vue 'v_clean_sondage' finalisée.")

    # ##########################################################################
    def ____BLOC_INGESTION(self): pass

    def _ingest_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        Método genérico privado para cargar un DF en Postgres.
        Usa DROP TABLE ... CASCADE para evitar errores de dependencia.
        """
        try:
            # 1. Normalizar nombres de columnas
            df.columns = [
                c.lower().strip().replace(' ', '_').replace('.', '_').replace("'", "_")
                for c in df.columns
            ]

            # 2. NIVEL EXPERTO: Borrado manual con CASCADE antes de to_sql
            # Esto elimina la tabla y las Vistas vinculadas (v_clean_...)
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))
                conn.commit()

            # 3. Ingestión (ahora if_exists='fail' o 'append' daría igual,
            # pero 'replace' es seguro porque ya no existe la tabla)
            df.to_sql(
                table_name,
                self.engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            print(f"✅ Tabla '{table_name}' cargada y dependencias limpiadas.")

        except Exception as e:
            print(f"❌ Error al cargar {table_name}: {e}")
            raise e

    def ingest_sirh(self, df: pd.DataFrame):
        # A. Normalización SIRH
        df['emp_id'] = df['id_employee'].astype(str).str.strip()
        self._ingest_dataframe(df, 'raw_sirh')

    def ingest_evals(self, df: pd.DataFrame):
        # B. Normalización EVALS (Quitar prefijo 'E_')
        df['emp_id'] = df['eval_number'].astype(str).str.replace('E_', '', regex=False).str.strip()
        self._ingest_dataframe(df, 'raw_evals')

    def ingest_sondage(self, df: pd.DataFrame):
        # C. Normalización SONDAGE (Quitar ceros a la izquierda)
        # Usamos fillna('') por seguridad antes de strip
        df['emp_id'] = df['code_sondage'].astype(str).str.lstrip('0').str.strip()
        self._ingest_dataframe(df, 'raw_sondage')

    # ##########################################################################
    def ____VIEW_MASTER(self): pass

# ##########################################################################
    # BLOC : GÉNÉRATION DE LA VUE MAÎTRESSE (MASTER VIEW)
    # ##########################################################################
    # Objectif : Fusionner SIRH, EVALS et SONDAGE sans perte de colonnes.
    # ##########################################################################

    def create_master_view(self):
        """
        Crée la vue v_master_clean en unifiant les trois sources de données.
        Garantit la disponibilité de chaque variable pour le Feature Engineering.
        """
        sql = r"""
        DROP VIEW IF EXISTS v_master_clean CASCADE;

        CREATE VIEW v_master_clean AS
        SELECT 
            -- -----------------------------------------------------------------
            -- 1. IDENTIFICATEURS (SIRH)
            -- -----------------------------------------------------------------
            s.emp_id,                         -- Clé unique de jointure
            s.id_employee,                    -- ID technique original

            -- -----------------------------------------------------------------
            -- 2. DONNÉES DÉMOGRAPHIQUES ET SOCIO-PRO (v_clean_sirh)
            -- -----------------------------------------------------------------
            s.age,                            -- Âge de l'employé
            s.genre,                          -- Genre (Masculin/Féminin)
            s.statut_marital,                 -- Situation familiale
            s.revenu_mensuel,                 -- Salaire mensuel brut
            s.departement,                    -- Direction/Département
            s.poste,                          -- Intitulé du poste
            s.nombre_experiences_precedentes, -- Nb entreprises avant TechNova
            s.nombre_heures_travailless,      -- Base horaire contrat
            s.annee_experience_totale,        -- Total années de carrière
            s.annees_dans_l_entreprise,       -- Ancienneté chez TechNova
            s.annees_dans_le_poste_actuel,    -- Ancienneté poste actuel

            -- -----------------------------------------------------------------
            -- 3. PERFORMANCE ET SATISFACTION (v_clean_evals)
            -- -----------------------------------------------------------------
            e.satisfaction_employee_environnement,       -- Note environnement
            e.satisfaction_employee_nature_travail,      -- Note mission
            e.satisfaction_employee_equipe,              -- Note équipe
            e.satisfaction_employee_equilibre_pro_perso, -- Note WLB
            e.note_evaluation_actuelle,                  -- Performance N
            e.note_evaluation_precedente,                -- Performance N-1
            e.niveau_hierarchique_poste,                 -- Grade manager
            e.heure_supplementaires,                     -- Flag 0/1
            e.augementation_salaire_precedente,          -- % Augmentation

            -- -----------------------------------------------------------------
            -- 4. SONDAGE, CARRIÈRE ET CIBLE (v_clean_sondage)
            -- -----------------------------------------------------------------
            so.target_attrition,              -- VARIABLE CIBLE (0/1)
            so.nombre_participation_pee,      -- Plans épargne entreprise
            so.nb_formations_suivies,         -- Formations an dernier
            so.nombre_employee_sous_responsabilite, -- Taille équipe managée
            so.distance_domicile_travail,     -- KM domicile-travail
            so.niveau_education,              -- Niveau d'études (numérique)
            so.domaine_etude,                 -- Filière académique
            so.ayant_enfants,                 -- Charge de famille (0/1)
            so.frequence_deplacement,         -- Rythme de voyages pro
            so.annees_depuis_la_derniere_promotion, -- Stagnation promotionnelle
            so.annes_sous_responsable_actuel        -- Stabilité management
               

        FROM v_clean_sirh s
        INNER JOIN v_clean_evals   e  ON s.emp_id = e.emp_id
        INNER JOIN v_clean_sondage so ON s.emp_id = so.emp_id;
        """
        self._execute(sql, "Vue Maîtresse 'v_master_clean' générée (Jointure complète de 32 colonnes).")

    def get_master_data(self):
        """Carga la vista maestra en un DataFrame de Pandas."""
        query = "SELECT * FROM v_master_clean;"
        df = pd.read_sql(query, self.engine)
        # Actualizamos el estado interno de la clase padre
        self.df = df
        return df


    # ##########################################################################
    # --- FEATURE ENGINEERING ---
    # ##########################################################################
    def ____VIEW_FEATURES(self): pass

    def create_features_view(self):
        """
        Calcula indicadores clave (KPIs) de RRHH directamente en la base de datos.
        Basado en la vista v_master_clean.
        """
        sql = r"""
        DROP VIEW IF EXISTS v_features_engineering CASCADE;

        CREATE VIEW v_features_engineering AS
        SELECT 
            *,
            
            -- FE1: Ratio de stagnation (años sin promoción vs años en empresa)
            ROUND(
                (CAST(annees_depuis_la_derniere_promotion AS FLOAT) / 
                NULLIF(annees_dans_l_entreprise + 1, 0))::NUMERIC, 
                4
            ) AS fe1_ratio_stagnation,
            
            -- FE2: Stabilité Manager (tiempo con jefe actual vs tiempo en el puesto)
            ROUND(
                (CAST(annes_sous_responsable_actuel AS FLOAT) / 
                NULLIF(annees_dans_le_poste_actuel + 1, 0))::NUMERIC, 
                4
            ) AS fe2_stabilite_manager,
            
            -- FE3: Indice Job Hopping (experiencia total vs número de empresas)
            ROUND(
                (CAST(annee_experience_totale AS FLOAT) / 
                NULLIF(nombre_experiences_precedentes + 1, 0))::NUMERIC, 
                4
            ) AS fe3_indice_job_hopping,
            
            -- FE4: Ancienneté relative (porcentaje de vida adulta en la empresa)
            ROUND(
                (CAST(annees_dans_l_entreprise AS FLOAT) / 
                NULLIF(GREATEST(age - 18, 1), 0))::NUMERIC, 
                4
            ) AS fe4_anciennete_relative,
            
            -- FE5: Satisfaction globale (Promedio de métricas de satisfacción)
            ROUND(
                ((satisfaction_employee_environnement + 
                 satisfaction_employee_nature_travail + 
                 satisfaction_employee_equipe + 
                 satisfaction_employee_equilibre_pro_perso) / 4.0)::NUMERIC, 
                2
            ) AS fe5_satisfaction_globale,
            
            -- FE6: Risque d'overwork (horas extra penalizadas por falta de equilibrio)
            ROUND(
                (heure_supplementaires * (1.0 / NULLIF(satisfaction_employee_equilibre_pro_perso + 1, 0)))::NUMERIC, 
                4
            ) AS fe6_risque_overwork,
            
            -- FE7: Pénibilité trajet (impacto de distancia si además hace horas extra)
            heure_supplementaires * distance_domicile_travail AS fe7_penibilite_trajet,
            
            -- FE8: Valeur de l'expérience (salario por cada año de experiencia total)
            ROUND(
                (CAST(revenu_mensuel AS FLOAT) / 
                NULLIF(annee_experience_totale + 1, 0))::NUMERIC, 
                2
            ) AS fe8_valeur_experience

        FROM v_master_clean;
        """
        self._execute(sql, "Vista de Ingeniería de Características 'v_features_engineering' creada.")

    def get_features_data(self):
        """Retorna el DataFrame final con todas las features calculadas."""
        query = "SELECT * FROM v_features_engineering;"
        df = pd.read_sql(query, self.engine)
        self.df = df # Actualizamos el estado para que DataCleaner.df tenga las features
        return df
