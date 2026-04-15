"""
data/schema.py — Registre d'attributs (dataclasses, sans Pydantic)
===============================================================

Objectif :
- Centraliser le dictionnaire métier ↔ technique (noms + valeurs).
- Dériver automatiquement les listes de colonnes (OHE, log, standard, etc.).
"""

from __future__ import annotations

# ==============================================================================
# IMPORTS - LIBRAIRIES STANDARD PYTHON
# ==============================================================================
import json                               # Sérialisation (export JSON)

# ==============================================================================
# IMPORTS - TYPAGE, DATACLASSES, ENUMS
# ==============================================================================
from   dataclasses import dataclass, field  # Modèles simples et immuables
from   enum        import Enum              # Valeurs contrôlées (type/role)
from   typing      import Dict, List, Optional  # Typage lisible


# ##############################################################################
# ENUMS - TYPES, RÔLES ET TRANSFORMATIONS
# ##############################################################################


class ColumnType(str, Enum):
    """Type logique de colonne (métier)."""

    CATEGORICAL = "categorical"
    NUMERICAL   = "numerical"
    BINARY      = "binary"
    DATETIME    = "datetime"
    IDENTIFIER  = "identifier"


class ColumnRole(str, Enum):
    """Rôle d'une colonne dans le pipeline."""

    FEATURE     = "feature"
    TARGET      = "target"
    IDENTIFIER  = "identifier"
    DROP        = "drop"
    META        = "meta"


class EncodingType(str, Enum):
    """Stratégie d'encodage (surtout pour les catégorielles)."""

    ONE_HOT     = "one_hot"
    ORDINAL     = "ordinal"
    TARGET_ENC  = "TARGET_ENC"
    BINARY_ENC  = "binary"
    NONE        = "none"


class TransformType(str, Enum):
    """Transformation numérique (avant/pendant le preprocessing)."""

    LOG         = "log"
    STANDARD    = "standard"
    ROBUST      = "robust"
    NONE        = "none"


# ##############################################################################
# DATACLASS - SPÉCIFICATION D'UN ATTRIBUT
# ##############################################################################


@dataclass
class AttributeSpec:
    """
    Spécification d'un attribut du modèle.

    - name_raw       : nom tel que reçu (CSV / SQL raw)
    - name_metier    : libellé métier (lisible)
    - name_technique : nom stable pour le modèle / API (snake_case)
    - source_table   : table logique d'origine (application, bureau_agg, ...)
    """

    name_raw: str
    name_metier: str
    name_technique: str
    source_table: str

    description: str = ""

    col_type: ColumnType       = ColumnType.NUMERICAL
    role: ColumnRole           = ColumnRole.FEATURE
    encoding: EncodingType     = EncodingType.NONE
    transform: TransformType   = TransformType.NONE

    valeurs_possibles: Optional[Dict[str, str]] = field(default=None)
    valeur_inconnue: str = "unknown"

    learned_median: Optional[float]       = field(default=None)
    learned_mode: Optional[str]           = field(default=None)
    learned_winsor_low: Optional[float]   = field(default=None)
    learned_winsor_high: Optional[float]  = field(default=None)
    learned_log_decalage: Optional[float] = field(default=None)

    nullable: bool = True
    version: str = "1.0"

    def to_dict(self) -> Dict:
        """
        Exporte l'attribut en dictionnaire JSON-friendly.

        - Enum → .value
        - dict → JSON string (pour export YAML/DB plus simple)
        """
        return {
            key: (
                value.value
                if isinstance(value, Enum)
                else (json.dumps(value) if isinstance(value, dict) else value)
            )
            for key, value in self.__dict__.items()
        }


# ##############################################################################
# REGISTRE - ACCÈS ET DÉRIVATION DES LISTES DE COLONNES
# ##############################################################################


class FeatureRegistry:
    """Registre des attributs et helpers de dérivation (OHE, log, standard...)."""

    def __init__(self, attributes: List[AttributeSpec]):
        self.attributes = attributes
        self._idx: Dict[str, AttributeSpec] = {
            attr.name_technique: attr for attr in attributes
        }

    # -------------------------------------------------------------------------
    # Accès (lookup)
    # -------------------------------------------------------------------------

    def get_attr(self, name_technique: str) -> Optional[AttributeSpec]:
        """Retourne la spec d'un attribut à partir de son nom technique."""
        return self._idx.get(name_technique)

    def get_attr_by_raw(self, name_raw: str) -> Optional[AttributeSpec]:
        """Retourne la spec d'un attribut à partir de son nom raw."""
        return next(
            (attr for attr in self.attributes if attr.name_raw == name_raw),
            None,
        )

    # -------------------------------------------------------------------------
    # Dérivation de listes
    # -------------------------------------------------------------------------

    def get_columns(
        self,
        role: Optional[ColumnRole] = None,
        col_type: Optional[ColumnType] = None,
        encoding: Optional[EncodingType] = None,
        transform: Optional[TransformType] = None,
        present_in: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Filtre les attributs selon des critères et retourne les noms techniques.

        present_in : si fourni, restreint la liste aux colonnes présentes.
        """
        resultats = self.attributes

        if role is not None:
            resultats = [a for a in resultats if a.role == role]
        if col_type is not None:
            resultats = [a for a in resultats if a.col_type == col_type]
        if encoding is not None:
            resultats = [a for a in resultats if a.encoding == encoding]
        if transform is not None:
            resultats = [a for a in resultats if a.transform == transform]

        noms = [a.name_technique for a in resultats]

        if present_in is not None:
            colonnes_presentes = set(present_in)
            noms = [n for n in noms if n in colonnes_presentes]

        return noms

    # -------------------------------------------------------------------------
    # Propriétés usuelles (shortcuts)
    # -------------------------------------------------------------------------

    @property
    def cols_ohe(self) -> List[str]:
        return self.get_columns(encoding=EncodingType.ONE_HOT)

    @property
    def cols_ordinal(self) -> List[str]:
        return self.get_columns(encoding=EncodingType.ORDINAL)

    @property
    def cols_target_enc(self) -> List[str]:
        return self.get_columns(encoding=EncodingType.TARGET_ENC)

    @property
    def cols_log(self) -> List[str]:
        return self.get_columns(transform=TransformType.LOG)

    @property
    def cols_standard(self) -> List[str]:
        return self.get_columns(transform=TransformType.STANDARD)

    @property
    def cols_robust(self) -> List[str]:
        return self.get_columns(transform=TransformType.ROBUST)

    @property
    def cols_drop(self) -> List[str]:
        return self.get_columns(role=ColumnRole.DROP)

    @property
    def cols_identifiers(self) -> List[str]:
        return self.get_columns(role=ColumnRole.IDENTIFIER)

    @property
    def cols_features(self) -> List[str]:
        return self.get_columns(role=ColumnRole.FEATURE)

    @property
    def cols_categorical(self) -> List[str]:
        return self.get_columns(col_type=ColumnType.CATEGORICAL)

    @property
    def cols_numerical(self) -> List[str]:
        return self.get_columns(col_type=ColumnType.NUMERICAL)

    @property
    def cols_binary(self) -> List[str]:
        return self.get_columns(col_type=ColumnType.BINARY)

    @property
    def col_target(self) -> Optional[str]:
        cibles = self.get_columns(role=ColumnRole.TARGET)
        return cibles[0] if cibles else None

    @property
    def rename_map(self) -> Dict[str, str]:
        """Mapping name_raw → name_technique (si différents)."""
        return {
            a.name_raw: a.name_technique
            for a in self.attributes
            if a.name_raw != a.name_technique
        }

    # -------------------------------------------------------------------------
    # Valeurs et validation
    # -------------------------------------------------------------------------

    def get_value_mapping(self, name_technique: str) -> Optional[Dict[str, str]]:
        """Retourne le mapping de valeurs pour une colonne technique."""
        attr = self.get_attr(name_technique)
        return attr.valeurs_possibles if attr else None

    def validate_dataframe_BAK(self, df_columns: List[str]) -> Dict:
        """Valide un jeu de colonnes vs registre (inconnues, manquantes, ok)."""
        colonnes_connues = set(self._idx.keys())

        return {
            "unknown_in_df": [
                c for c in df_columns if c not in colonnes_connues
            ],
            "missing_from_df": [
                a.name_technique
                for a in self.attributes
                if a.name_technique not in df_columns
                and a.role == ColumnRole.FEATURE
            ],
            "ok": all(c in colonnes_connues for c in df_columns),
        }

    def validate_dataframe_robust(self, df_columns: List[str]) -> Dict:
        """
        Versión final y blindada:
        - Case-Insensitive total (compara 'A' con 'a' correctamente).
        - Excluye tablas RAW para evitar falsos positivos (las 77 columnas).
        - Compatible con el formato de retorno Dict {unknown, missing, ok}.
        """
        # 1. Fuentes a excluir (evitamos que las tablas RAW marquen como 'missing')
        exclude_sources = [
            "bureau", "bureau_balance", 
            "previous_application", 
            "pos_cash", "credit_card", "installments"
        ]

        # 2. Atributos relevantes del registro (filtrados)
        relevant_attrs = [
            a for a in self.attributes 
            if a.source_table not in exclude_sources
        ]

        # 3. Normalización del registro
        # Mapeo de { 'nombre_bajo': 'Nombre_Original' }
        registry_map_lower = {a.name_technique.lower(): a.name_technique for a in relevant_attrs}
        
        # 4. Normalización de las columnas del DataFrame (lo que recibimos)
        df_cols_lower = [c.lower() for c in df_columns]
        df_cols_set_lower = set(df_cols_lower)

        # 5. Cálculo de discrepancias
        # 5.1 Desconocidas: Están en el DF pero no en nuestro mapa de registro (ignorando 'split')
        unknown_in_df = [
            c for c in df_columns 
            if c.lower() not in registry_map_lower and c.lower() != 'split'
        ]
        
        # 5.2 Faltantes: Están en el registro pero NO están en el DF (comparación en minúsculas)
        missing_from_df = [
            orig_name for lower_name, orig_name in registry_map_lower.items() 
            if lower_name not in df_cols_set_lower
        ]

        # 6. Retorno estructurado
        return {
            "unknown_in_df": unknown_in_df,
            "missing_from_df": missing_from_df,
            "ok": len(missing_from_df) == 0  # OK si no falta ninguna columna necesaria
        }
    
    # -------------------------------------------------------------------------
    # Exports
    # -------------------------------------------------------------------------

    def to_json(self, path: str) -> None:
        """Exporte le registre en JSON (liste d'objets)."""
        with open(path, "w", encoding="utf-8") as fichier:
            json.dump(
                [a.to_dict() for a in self.attributes],
                fichier,
                indent=2,
                ensure_ascii=False,
            )

        print(f"✅ Registre exporté........: {path}")

    def to_yaml(self, path: str) -> None:
        """Exporte le registre en YAML (fallback JSON si PyYAML absent)."""
        try:
            import yaml  # type: ignore
        except ImportError:
            self.to_json(path.replace(".yaml", ".json"))
            return

        with open(path, "w", encoding="utf-8") as fichier:
            yaml.dump(
                [a.to_dict() for a in self.attributes],
                fichier,
                allow_unicode=True,
                sort_keys=False,
            )

    def sync_to_db(self, engine) -> None:
        """Synchronise le registre dans une table SQL `feature_registry`."""
        import pandas as pd

        rows = [a.to_dict() for a in self.attributes]
        pd.DataFrame(rows).to_sql(
            "feature_registry",
            engine,
            if_exists="replace",
            index=False,
        )
        print(f"✅ Registre synchronisé....: {len(rows)} attributs")

    # -------------------------------------------------------------------------
    # Rapport rapide
    # -------------------------------------------------------------------------

    def summary(self) -> None:
        """Affiche un résumé lisible du registre."""
        print("\n============================================================================")
        print("RAPPORT DU REGISTRE")
        print("============================================================================")
        print(f"  Attributs total........: {len(self.attributes)}")
        print(f"  Encodage One-Hot........: {len(self.cols_ohe)}")
        print(f"  Encodage ordinal........: {len(self.cols_ordinal)}")
        print(f"  Target encoding.........: {len(self.cols_target_enc)}")
        print(f"  Transform log...........: {len(self.cols_log)}")
        print(f"  StandardScaler..........: {len(self.cols_standard)}")
        print(f"  RobustScaler............: {len(self.cols_robust)}")


# ##############################################################################
# HELPER - FACTORY D'ATTRIBUTS
# ##############################################################################


def build_attribute_spec(
    name_raw: str,
    name_metier: str,
    name_technique: str,
    source_table: str = "application",
    col_type: ColumnType = ColumnType.NUMERICAL,
    role: ColumnRole = ColumnRole.FEATURE,
    encoding: EncodingType = EncodingType.NONE,
    transform: TransformType = TransformType.NONE,
    valeurs_possibles: Optional[Dict[str, str]] = None,
    valeur_inconnue: str = "unknown",
    description: str = "",
) -> AttributeSpec:
    """Factory courte pour créer une AttributeSpec avec valeurs par défaut."""
    return AttributeSpec(
        name_raw=name_raw,
        name_metier=name_metier,
        name_technique=name_technique,
        source_table=source_table,
        description=description,
        col_type=col_type,
        role=role,
        encoding=encoding,
        transform=transform,
        valeurs_possibles=valeurs_possibles,
        valeur_inconnue=valeur_inconnue,
    )


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY BOOTSTRAP — généré automatiquement, à réviser avant usage
# ═══════════════════════════════════════════════════════════════════════

REGISTRY = FeatureRegistry(
    attributes=[

        build_attribute_spec(
            "split",
            "Split train/test",
            "split",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            role=ColumnRole.META,
            encoding=EncodingType.NONE,
        ),
        # ════════════════════════════════════════════════════════════
        # APPLICATION
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_CURR",
            "ID of loan in our",  # TODO: réviser name_metier
            "sk_id_curr",
            source_table="application",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "TARGET",
            "Target variable",  # TODO: réviser name_metier
            "target",
            source_table="application",
            role=ColumnRole.TARGET,
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "NAME_CONTRACT_TYPE",
            "Contract product type of the",  # TODO: réviser name_metier
            "name_contract_type",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Cash loans": "cash_loans",
                "Revolving loans": "revolving_loans",
            },
        ),  # card=2

        build_attribute_spec(
            "CODE_GENDER",
            "Gender of the client",  # TODO: réviser name_metier
            "code_gender",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "F": "f",
                "M": "m",
                "XNA": "xna",
            },
        ),  # card=3

        build_attribute_spec(
            "FLAG_OWN_CAR",
            "Flag if the client owns",  # TODO: réviser name_metier
            "flag_own_car",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "N": "n",
                "Y": "y",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_OWN_REALTY",
            "Flag if client owns a",  # TODO: réviser name_metier
            "flag_own_realty",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "N": "n",
                "Y": "y",
            },
        ),  # card=2

        build_attribute_spec(
            "CNT_CHILDREN",
            "Number of children the client",  # TODO: réviser name_metier
            "cnt_children",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=11.0

        build_attribute_spec(
            "AMT_INCOME_TOTAL",
            "Income of the client",  # TODO: réviser name_metier
            "amt_income_total",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=25650.0, max=117000000.0

        build_attribute_spec(
            "AMT_CREDIT",
            "Final credit amount on the",  # TODO: réviser name_metier
            "amt_credit",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=45000.0, max=4050000.0

        build_attribute_spec(
            "AMT_ANNUITY",
            "Annuity of previous application",  # TODO: réviser name_metier
            "amt_annuity",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=2052.0, max=258025.5

        build_attribute_spec(
            "AMT_GOODS_PRICE",
            "Goods price of good that",  # TODO: réviser name_metier
            "amt_goods_price",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=45000.0, max=4050000.0

        build_attribute_spec(
            "NAME_TYPE_SUITE",
            "Who accompanied client when applying",  # TODO: réviser name_metier
            "name_type_suite",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Children": "children",
                "Family": "family",
                "Group of people": "group_of_people",
                "Other_A": "other_a",
                "Other_B": "other_b",
                "Spouse, partner": "spouse_partner",
                "Unaccompanied": "unaccompanied",
            },
        ),  # card=7

        build_attribute_spec(
            "NAME_INCOME_TYPE",
            "Clients income type",  # TODO: réviser name_metier
            "name_income_type",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Businessman": "businessman",
                "Commercial associate": "commercial_associate",
                "Maternity leave": "maternity_leave",
                "Pensioner": "pensioner",
                "State servant": "state_servant",
                "Student": "student",
                "Unemployed": "unemployed",
                "Working": "working",
            },
        ),  # card=8

        build_attribute_spec(
            "NAME_EDUCATION_TYPE",
            "Level of highest education the",  # TODO: réviser name_metier
            "name_education_type",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Academic degree": "academic_degree",
                "Higher education": "higher_education",
                "Incomplete higher": "incomplete_higher",
                "Lower secondary": "lower_secondary",
                "Secondary / secondary special": "secondary_secondary_special",
            },
        ),  # card=5

        build_attribute_spec(
            "NAME_FAMILY_STATUS",
            "Family status of the client",  # TODO: réviser name_metier
            "name_family_status",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Civil marriage": "civil_marriage",
                "Married": "married",
                "Separated": "separated",
                "Single / not married": "single_not_married",
                "Unknown": "unknown",
                "Widow": "widow",
            },
        ),  # card=6

        build_attribute_spec(
            "NAME_HOUSING_TYPE",
            "What is the housing situation",  # TODO: réviser name_metier
            "name_housing_type",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Co-op apartment": "co_op_apartment",
                "House / apartment": "house_apartment",
                "Municipal apartment": "municipal_apartment",
                "Office apartment": "office_apartment",
                "Rented apartment": "rented_apartment",
                "With parents": "with_parents",
            },
        ),  # card=6

        build_attribute_spec(
            "REGION_POPULATION_RELATIVE",
            "Normalized population of region where",  # TODO: réviser name_metier
            "region_population_relative",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=0.07

        build_attribute_spec(
            "DAYS_BIRTH",
            "Client's age in days at",  # TODO: réviser name_metier
            "days_birth",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=-25184.0, max=-7680.0

        build_attribute_spec(
            "DAYS_EMPLOYED",
            "How many days before the",  # TODO: réviser name_metier
            "days_employed",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=-17531.0, max=365243.0

        build_attribute_spec(
            "DAYS_REGISTRATION",
            "How many days before the",  # TODO: réviser name_metier
            "days_registration",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=-22392.0, max=0.0

        build_attribute_spec(
            "DAYS_ID_PUBLISH",
            "How many days before the",  # TODO: réviser name_metier
            "days_id_publish",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=-6232.0, max=0.0

        build_attribute_spec(
            "OWN_CAR_AGE",
            "Age of client's car",  # TODO: réviser name_metier
            "own_car_age",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=65.0 | ⚠️ 66% null

        build_attribute_spec(
            "FLAG_MOBIL",
            "Did client provide mobile phone",  # TODO: réviser name_metier
            "flag_mobil",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_EMP_PHONE",
            "Did client provide work phone",  # TODO: réviser name_metier
            "flag_emp_phone",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_WORK_PHONE",
            "Did client provide home phone",  # TODO: réviser name_metier
            "flag_work_phone",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_CONT_MOBILE",
            "Was mobile phone reachable",  # TODO: réviser name_metier
            "flag_cont_mobile",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_PHONE",
            "Did client provide home phone",  # TODO: réviser name_metier
            "flag_phone",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_EMAIL",
            "Did client provide email",  # TODO: réviser name_metier
            "flag_email",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "OCCUPATION_TYPE",
            "What kind of occupation does",  # TODO: réviser name_metier
            "occupation_type",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ORDINAL,
        ),  # card=18 | ⚠️ 31% null

        build_attribute_spec(
            "CNT_FAM_MEMBERS",
            "How many family members does",  # TODO: réviser name_metier
            "cnt_fam_members",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=1.0, max=13.0

        build_attribute_spec(
            "REGION_RATING_CLIENT",
            "Our rating of the region",  # TODO: réviser name_metier
            "region_rating_client",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=1.0, max=3.0

        build_attribute_spec(
            "REGION_RATING_CLIENT_W_CITY",
            "Our rating of the region",  # TODO: réviser name_metier
            "region_rating_client_w_city",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=1.0, max=3.0

        build_attribute_spec(
            "WEEKDAY_APPR_PROCESS_START",
            "On which day of the",  # TODO: réviser name_metier
            "weekday_appr_process_start",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "FRIDAY": "friday",
                "MONDAY": "monday",
                "SATURDAY": "saturday",
                "SUNDAY": "sunday",
                "THURSDAY": "thursday",
                "TUESDAY": "tuesday",
                "WEDNESDAY": "wednesday",
            },
        ),  # card=7

        build_attribute_spec(
            "HOUR_APPR_PROCESS_START",
            "Approximately at what day hour",  # TODO: réviser name_metier
            "hour_appr_process_start",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=23.0

        build_attribute_spec(
            "REG_REGION_NOT_LIVE_REGION",
            "Flag if client's permanent address",  # TODO: réviser name_metier
            "reg_region_not_live_region",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0

        build_attribute_spec(
            "REG_REGION_NOT_WORK_REGION",
            "Flag if client's permanent address",  # TODO: réviser name_metier
            "reg_region_not_work_region",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0

        build_attribute_spec(
            "LIVE_REGION_NOT_WORK_REGION",
            "Flag if client's contact address",  # TODO: réviser name_metier
            "live_region_not_work_region",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0

        build_attribute_spec(
            "REG_CITY_NOT_LIVE_CITY",
            "Flag if client's permanent address",  # TODO: réviser name_metier
            "reg_city_not_live_city",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0

        build_attribute_spec(
            "REG_CITY_NOT_WORK_CITY",
            "Flag if client's permanent address",  # TODO: réviser name_metier
            "reg_city_not_work_city",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0

        build_attribute_spec(
            "LIVE_CITY_NOT_WORK_CITY",
            "Flag if client's contact address",  # TODO: réviser name_metier
            "live_city_not_work_city",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0

        build_attribute_spec(
            "ORGANIZATION_TYPE",
            "Type of organization where client",  # TODO: réviser name_metier
            "organization_type",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.TARGET_ENC,
        ),  # card=58

        build_attribute_spec(
            "EXT_SOURCE_1",
            "Normalized score from external data",  # TODO: réviser name_metier
            "ext_source_1",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.01, max=0.94 | ⚠️ 56% null

        build_attribute_spec(
            "EXT_SOURCE_2",
            "Normalized score from external data",  # TODO: réviser name_metier
            "ext_source_2",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=0.85

        build_attribute_spec(
            "EXT_SOURCE_3",
            "Normalized score from external data",  # TODO: réviser name_metier
            "ext_source_3",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=0.9

        build_attribute_spec(
            "APARTMENTS_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "apartments_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 51% null

        build_attribute_spec(
            "BASEMENTAREA_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "basementarea_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 58% null

        build_attribute_spec(
            "YEARS_BEGINEXPLUATATION_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "years_beginexpluatation_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 49% null

        build_attribute_spec(
            "YEARS_BUILD_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "years_build_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 66% null

        build_attribute_spec(
            "COMMONAREA_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "commonarea_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 70% null

        build_attribute_spec(
            "ELEVATORS_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "elevators_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 53% null

        build_attribute_spec(
            "ENTRANCES_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "entrances_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "FLOORSMAX_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "floorsmax_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "FLOORSMIN_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "floorsmin_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 68% null

        build_attribute_spec(
            "LANDAREA_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "landarea_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 59% null

        build_attribute_spec(
            "LIVINGAPARTMENTS_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "livingapartments_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 68% null

        build_attribute_spec(
            "LIVINGAREA_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "livingarea_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "NONLIVINGAPARTMENTS_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "nonlivingapartments_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 69% null

        build_attribute_spec(
            "NONLIVINGAREA_AVG",
            "Normalized information about building...",  # TODO: réviser name_metier
            "nonlivingarea_avg",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 55% null

        build_attribute_spec(
            "APARTMENTS_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "apartments_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 51% null

        build_attribute_spec(
            "BASEMENTAREA_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "basementarea_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 58% null

        build_attribute_spec(
            "YEARS_BEGINEXPLUATATION_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "years_beginexpluatation_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 49% null

        build_attribute_spec(
            "YEARS_BUILD_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "years_build_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 66% null

        build_attribute_spec(
            "COMMONAREA_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "commonarea_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 70% null

        build_attribute_spec(
            "ELEVATORS_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "elevators_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 53% null

        build_attribute_spec(
            "ENTRANCES_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "entrances_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "FLOORSMAX_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "floorsmax_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "FLOORSMIN_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "floorsmin_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 68% null

        build_attribute_spec(
            "LANDAREA_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "landarea_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 59% null

        build_attribute_spec(
            "LIVINGAPARTMENTS_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "livingapartments_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 68% null

        build_attribute_spec(
            "LIVINGAREA_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "livingarea_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "NONLIVINGAPARTMENTS_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "nonlivingapartments_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 69% null

        build_attribute_spec(
            "NONLIVINGAREA_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "nonlivingarea_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 55% null

        build_attribute_spec(
            "APARTMENTS_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "apartments_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 51% null

        build_attribute_spec(
            "BASEMENTAREA_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "basementarea_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 58% null

        build_attribute_spec(
            "YEARS_BEGINEXPLUATATION_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "years_beginexpluatation_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 49% null

        build_attribute_spec(
            "YEARS_BUILD_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "years_build_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 66% null

        build_attribute_spec(
            "COMMONAREA_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "commonarea_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 70% null

        build_attribute_spec(
            "ELEVATORS_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "elevators_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 53% null

        build_attribute_spec(
            "ENTRANCES_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "entrances_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "FLOORSMAX_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "floorsmax_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "FLOORSMIN_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "floorsmin_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 68% null

        build_attribute_spec(
            "LANDAREA_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "landarea_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 59% null

        build_attribute_spec(
            "LIVINGAPARTMENTS_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "livingapartments_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 68% null

        build_attribute_spec(
            "LIVINGAREA_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "livingarea_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 50% null

        build_attribute_spec(
            "NONLIVINGAPARTMENTS_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "nonlivingapartments_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 69% null

        build_attribute_spec(
            "NONLIVINGAREA_MEDI",
            "Normalized information about building...",  # TODO: réviser name_metier
            "nonlivingarea_medi",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 55% null

        build_attribute_spec(
            "FONDKAPREMONT_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "fondkapremont_mode",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "not specified": "not_specified",
                "org spec account": "org_spec_account",
                "reg oper account": "reg_oper_account",
                "reg oper spec account": "reg_oper_spec_account",
            },
        ),  # card=4 | ⚠️ 68% null

        build_attribute_spec(
            "HOUSETYPE_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "housetype_mode",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "block of flats": "block_of_flats",
                "specific housing": "specific_housing",
                "terraced house": "terraced_house",
            },
        ),  # card=3 | ⚠️ 50% null

        build_attribute_spec(
            "TOTALAREA_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "totalarea_mode",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1.0 | ⚠️ 48% null

        build_attribute_spec(
            "WALLSMATERIAL_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "wallsmaterial_mode",
            source_table="application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Block": "block",
                "Mixed": "mixed",
                "Monolithic": "monolithic",
                "Others": "others",
                "Panel": "panel",
                "Stone, brick": "stone_brick",
                "Wooden": "wooden",
            },
        ),  # card=7 | ⚠️ 51% null

        build_attribute_spec(
            "EMERGENCYSTATE_MODE",
            "Normalized information about building...",  # TODO: réviser name_metier
            "emergencystate_mode",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "No": "no",
                "Yes": "yes",
            },
        ),  # card=2 | ⚠️ 47% null

        build_attribute_spec(
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "How many observation of client's",  # TODO: réviser name_metier
            "obs_30_cnt_social_circle",
            source_table="application",
            transform=TransformType.ROBUST,
        ),  # min=0.0, max=28.0

        build_attribute_spec(
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "How many observation of client's",  # TODO: réviser name_metier
            "def_30_cnt_social_circle",
            source_table="application",
            transform=TransformType.ROBUST,
        ),  # min=0.0, max=6.0

        build_attribute_spec(
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "How many observation of client's",  # TODO: réviser name_metier
            "obs_60_cnt_social_circle",
            source_table="application",
            transform=TransformType.ROBUST,
        ),  # min=0.0, max=28.0

        build_attribute_spec(
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "How many observation of client's",  # TODO: réviser name_metier
            "def_60_cnt_social_circle",
            source_table="application",
            transform=TransformType.ROBUST,
        ),  # min=0.0, max=5.0

        build_attribute_spec(
            "DAYS_LAST_PHONE_CHANGE",
            "How many days before application",  # TODO: réviser name_metier
            "days_last_phone_change",
            source_table="application",
            transform=TransformType.STANDARD,
        ),  # min=-4002.0, max=0.0

        build_attribute_spec(
            "FLAG_DOCUMENT_2",
            "Did client provide document 2",  # TODO: réviser name_metier
            "flag_document_2",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_3",
            "Did client provide document 3",  # TODO: réviser name_metier
            "flag_document_3",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_4",
            "Did client provide document 4",  # TODO: réviser name_metier
            "flag_document_4",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_5",
            "Did client provide document 5",  # TODO: réviser name_metier
            "flag_document_5",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_6",
            "Did client provide document 6",  # TODO: réviser name_metier
            "flag_document_6",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_7",
            "Did client provide document 7",  # TODO: réviser name_metier
            "flag_document_7",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_8",
            "Did client provide document 8",  # TODO: réviser name_metier
            "flag_document_8",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_9",
            "Did client provide document 9",  # TODO: réviser name_metier
            "flag_document_9",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_10",
            "Did client provide document 10",  # TODO: réviser name_metier
            "flag_document_10",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_11",
            "Did client provide document 11",  # TODO: réviser name_metier
            "flag_document_11",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_12",
            "Did client provide document 12",  # TODO: réviser name_metier
            "flag_document_12",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
            },
        ),  # card=1

        build_attribute_spec(
            "FLAG_DOCUMENT_13",
            "Did client provide document 13",  # TODO: réviser name_metier
            "flag_document_13",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_14",
            "Did client provide document 14",  # TODO: réviser name_metier
            "flag_document_14",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_15",
            "Did client provide document 15",  # TODO: réviser name_metier
            "flag_document_15",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_16",
            "Did client provide document 16",  # TODO: réviser name_metier
            "flag_document_16",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_17",
            "Did client provide document 17",  # TODO: réviser name_metier
            "flag_document_17",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_18",
            "Did client provide document 18",  # TODO: réviser name_metier
            "flag_document_18",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_19",
            "Did client provide document 19",  # TODO: réviser name_metier
            "flag_document_19",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_20",
            "Did client provide document 20",  # TODO: réviser name_metier
            "flag_document_20",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "FLAG_DOCUMENT_21",
            "Did client provide document 21",  # TODO: réviser name_metier
            "flag_document_21",
            source_table="application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "Number of enquiries to Credit",  # TODO: réviser name_metier
            "amt_req_credit_bureau_hour",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=3.0

        build_attribute_spec(
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "Number of enquiries to Credit",  # TODO: réviser name_metier
            "amt_req_credit_bureau_day",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=6.0

        build_attribute_spec(
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "Number of enquiries to Credit",  # TODO: réviser name_metier
            "amt_req_credit_bureau_week",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=6.0

        build_attribute_spec(
            "AMT_REQ_CREDIT_BUREAU_MON",
            "Number of enquiries to Credit",  # TODO: réviser name_metier
            "amt_req_credit_bureau_mon",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=24.0

        build_attribute_spec(
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "Number of enquiries to Credit",  # TODO: réviser name_metier
            "amt_req_credit_bureau_qrt",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=8.0

        build_attribute_spec(
            "AMT_REQ_CREDIT_BUREAU_YEAR",
            "Number of enquiries to Credit",  # TODO: réviser name_metier
            "amt_req_credit_bureau_year",
            source_table="application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=25.0


        # ════════════════════════════════════════════════════════════
        # BUREAU
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_CURR",
            "ID of loan in our",  # TODO: réviser name_metier
            "sk_id_curr",
            source_table="bureau",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "SK_ID_BUREAU",
            "SK_ID_BUREAU",  # TODO: réviser name_metier
            "sk_id_bureau",
            source_table="bureau",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "CREDIT_ACTIVE",
            "Status of the Credit Bureau",  # TODO: réviser name_metier
            "credit_active",
            source_table="bureau",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Active": "active",
                "Bad debt": "bad_debt",
                "Closed": "closed",
                "Sold": "sold",
            },
        ),  # card=4

        build_attribute_spec(
            "CREDIT_CURRENCY",
            "Recoded currency of the Credit",  # TODO: réviser name_metier
            "credit_currency",
            source_table="bureau",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "currency 1": "currency_1",
                "currency 2": "currency_2",
                "currency 3": "currency_3",
                "currency 4": "currency_4",
            },
        ),  # card=4

        build_attribute_spec(
            "DAYS_CREDIT",
            "How many days before current",  # TODO: réviser name_metier
            "days_credit",
            source_table="bureau",
            transform=TransformType.STANDARD,
        ),  # min=-2922.0, max=-2.0

        build_attribute_spec(
            "CREDIT_DAY_OVERDUE",
            "Number of days past due",  # TODO: réviser name_metier
            "credit_day_overdue",
            source_table="bureau",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=2625.0

        build_attribute_spec(
            "DAYS_CREDIT_ENDDATE",
            "Remaining duration of CB credit",  # TODO: réviser name_metier
            "days_credit_enddate",
            source_table="bureau",
            transform=TransformType.STANDARD,
        ),  # min=-2873.0, max=31198.0

        build_attribute_spec(
            "DAYS_ENDDATE_FACT",
            "Days since CB credit ended",  # TODO: réviser name_metier
            "days_enddate_fact",
            source_table="bureau",
            transform=TransformType.STANDARD,
        ),  # min=-2873.0, max=0.0 | ⚠️ 38% null

        build_attribute_spec(
            "AMT_CREDIT_MAX_OVERDUE",
            "Maximal amount overdue on the",  # TODO: réviser name_metier
            "amt_credit_max_overdue",
            source_table="bureau",
            transform=TransformType.LOG,
        ),  # min=0.0, max=10861812.0 | ⚠️ 64% null

        build_attribute_spec(
            "CNT_CREDIT_PROLONG",
            "How many times was the",  # TODO: réviser name_metier
            "cnt_credit_prolong",
            source_table="bureau",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=4.0

        build_attribute_spec(
            "AMT_CREDIT_SUM",
            "Current credit amount for the",  # TODO: réviser name_metier
            "amt_credit_sum",
            source_table="bureau",
            transform=TransformType.LOG,
        ),  # min=0.0, max=45000000.0

        build_attribute_spec(
            "AMT_CREDIT_SUM_DEBT",
            "Current debt on Credit Bureau",  # TODO: réviser name_metier
            "amt_credit_sum_debt",
            source_table="bureau",
            transform=TransformType.LOG,
        ),  # min=-901224.27, max=22410000.0

        build_attribute_spec(
            "AMT_CREDIT_SUM_LIMIT",
            "Current credit limit of credit",  # TODO: réviser name_metier
            "amt_credit_sum_limit",
            source_table="bureau",
            transform=TransformType.LOG,
        ),  # min=-110293.6, max=1352044.84 | ⚠️ 34% null

        build_attribute_spec(
            "AMT_CREDIT_SUM_OVERDUE",
            "Current amount overdue on Credit",  # TODO: réviser name_metier
            "amt_credit_sum_overdue",
            source_table="bureau",
            transform=TransformType.LOG,
        ),  # min=0.0, max=349428.06

        build_attribute_spec(
            "CREDIT_TYPE",
            "Type of Credit Bureau credit",  # TODO: réviser name_metier
            "credit_type",
            source_table="bureau",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Another type of loan": "another_type_of_loan",
                "Car loan": "car_loan",
                "Consumer credit": "consumer_credit",
                "Credit card": "credit_card",
                "Loan for business development": "loan_for_business_development",
                "Loan for working capital replenishment": "loan_for_working_capital_replenishment",
                "Microloan": "microloan",
                "Mortgage": "mortgage",
                "Real estate loan": "real_estate_loan",
                "Unknown type of loan": "unknown_type_of_loan",
            },
        ),  # card=10

        build_attribute_spec(
            "DAYS_CREDIT_UPDATE",
            "How many days before loan",  # TODO: réviser name_metier
            "days_credit_update",
            source_table="bureau",
            transform=TransformType.STANDARD,
        ),  # min=-41913.0, max=0.0

        build_attribute_spec(
            "AMT_ANNUITY",
            "Annuity of previous application",  # TODO: réviser name_metier
            "amt_annuity",
            source_table="bureau",
            transform=TransformType.LOG,
        ),  # min=0.0, max=6578707.5 | ⚠️ 80% null


        # ════════════════════════════════════════════════════════════
        # BUREAU BALANCE
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_BUREAU",
            "SK_ID_BUREAU",  # TODO: réviser name_metier
            "sk_id_bureau",
            source_table="bureau_balance",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "MONTHS_BALANCE",
            "Month of balance relative to",  # TODO: réviser name_metier
            "months_balance",
            source_table="bureau_balance",
            transform=TransformType.STANDARD,
        ),  # min=-96.0, max=0.0

        build_attribute_spec(
            "STATUS",
            "Status of Credit Bureau loan",  # TODO: réviser name_metier
            "status",
            source_table="bureau_balance",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
                "2": "v_2",
                "3": "v_3",
                "4": "v_4",
                "5": "v_5",
                "C": "c",
                "X": "x",
            },
        ),  # card=8


        # ════════════════════════════════════════════════════════════
        # PREVIOUS APPLICATION
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_PREV",
            "ID of previous credit in",  # TODO: réviser name_metier
            "sk_id_prev",
            source_table="previous_application",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "SK_ID_CURR",
            "ID of loan in our",  # TODO: réviser name_metier
            "sk_id_curr",
            source_table="previous_application",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "NAME_CONTRACT_TYPE",
            "Contract product type of the",  # TODO: réviser name_metier
            "name_contract_type",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Cash loans": "cash_loans",
                "Consumer loans": "consumer_loans",
                "Revolving loans": "revolving_loans",
                "XNA": "xna",
            },
        ),  # card=4

        build_attribute_spec(
            "AMT_ANNUITY",
            "Annuity of previous application",  # TODO: réviser name_metier
            "amt_annuity",
            source_table="previous_application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=234478.39

        build_attribute_spec(
            "AMT_APPLICATION",
            "For how much credit did",  # TODO: réviser name_metier
            "amt_application",
            source_table="previous_application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=3826372.5

        build_attribute_spec(
            "AMT_CREDIT",
            "Final credit amount on the",  # TODO: réviser name_metier
            "amt_credit",
            source_table="previous_application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=4104351.0

        build_attribute_spec(
            "AMT_DOWN_PAYMENT",
            "Down payment on the previous",  # TODO: réviser name_metier
            "amt_down_payment",
            source_table="previous_application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=1035000.0 | ⚠️ 50% null

        build_attribute_spec(
            "AMT_GOODS_PRICE",
            "Goods price of good that",  # TODO: réviser name_metier
            "amt_goods_price",
            source_table="previous_application",
            transform=TransformType.LOG,
        ),  # min=0.0, max=3826372.5

        build_attribute_spec(
            "WEEKDAY_APPR_PROCESS_START",
            "On which day of the",  # TODO: réviser name_metier
            "weekday_appr_process_start",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "FRIDAY": "friday",
                "MONDAY": "monday",
                "SATURDAY": "saturday",
                "SUNDAY": "sunday",
                "THURSDAY": "thursday",
                "TUESDAY": "tuesday",
                "WEDNESDAY": "wednesday",
            },
        ),  # card=7

        build_attribute_spec(
            "HOUR_APPR_PROCESS_START",
            "Approximately at what day hour",  # TODO: réviser name_metier
            "hour_appr_process_start",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=23.0

        build_attribute_spec(
            "FLAG_LAST_APPL_PER_CONTRACT",
            "Flag if it was last",  # TODO: réviser name_metier
            "flag_last_appl_per_contract",
            source_table="previous_application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "N": "n",
                "Y": "y",
            },
        ),  # card=2

        build_attribute_spec(
            "NFLAG_LAST_APPL_IN_DAY",
            "Flag if the application was",  # TODO: réviser name_metier
            "nflag_last_appl_in_day",
            source_table="previous_application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0": "v_0",
                "1": "v_1",
            },
        ),  # card=2

        build_attribute_spec(
            "RATE_DOWN_PAYMENT",
            "Down payment rate normalized on",  # TODO: réviser name_metier
            "rate_down_payment",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=0.94 | ⚠️ 50% null

        build_attribute_spec(
            "RATE_INTEREST_PRIMARY",
            "Interest rate normalized on previous",  # TODO: réviser name_metier
            "rate_interest_primary",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=0.06, max=0.7 | ⚠️ 100% null

        build_attribute_spec(
            "RATE_INTEREST_PRIVILEGED",
            "Interest rate normalized on previous",  # TODO: réviser name_metier
            "rate_interest_privileged",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=0.42, max=0.87 | ⚠️ 100% null

        build_attribute_spec(
            "NAME_CASH_LOAN_PURPOSE",
            "Purpose of the cash loan",  # TODO: réviser name_metier
            "name_cash_loan_purpose",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ORDINAL,
        ),  # card=24

        build_attribute_spec(
            "NAME_CONTRACT_STATUS",
            "Contract status of previous application",  # TODO: réviser name_metier
            "name_contract_status",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Approved": "approved",
                "Canceled": "canceled",
                "Refused": "refused",
                "Unused offer": "unused_offer",
            },
        ),  # card=4

        build_attribute_spec(
            "DAYS_DECISION",
            "Relative to current application when",  # TODO: réviser name_metier
            "days_decision",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-2922.0, max=-2.0

        build_attribute_spec(
            "NAME_PAYMENT_TYPE",
            "Payment method that client chose",  # TODO: réviser name_metier
            "name_payment_type",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Cash through the bank": "cash_through_the_bank",
                "Cashless from the account of the employer": "cashless_from_the_account_of_the_employer",
                "Non-cash from your account": "non_cash_from_your_account",
                "XNA": "xna",
            },
        ),  # card=4

        build_attribute_spec(
            "CODE_REJECT_REASON",
            "Why was the previous application",  # TODO: réviser name_metier
            "code_reject_reason",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "CLIENT": "client",
                "HC": "hc",
                "LIMIT": "limit",
                "SCO": "sco",
                "SCOFR": "scofr",
                "SYSTEM": "system",
                "VERIF": "verif",
                "XAP": "xap",
                "XNA": "xna",
            },
        ),  # card=9

        build_attribute_spec(
            "NAME_TYPE_SUITE",
            "Who accompanied client when applying",  # TODO: réviser name_metier
            "name_type_suite",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Children": "children",
                "Family": "family",
                "Group of people": "group_of_people",
                "Other_A": "other_a",
                "Other_B": "other_b",
                "Spouse, partner": "spouse_partner",
                "Unaccompanied": "unaccompanied",
            },
        ),  # card=7 | ⚠️ 48% null

        build_attribute_spec(
            "NAME_CLIENT_TYPE",
            "Was the client old or",  # TODO: réviser name_metier
            "name_client_type",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "New": "new",
                "Refreshed": "refreshed",
                "Repeater": "repeater",
                "XNA": "xna",
            },
        ),  # card=4

        build_attribute_spec(
            "NAME_GOODS_CATEGORY",
            "What kind of goods did",  # TODO: réviser name_metier
            "name_goods_category",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ORDINAL,
        ),  # card=26

        build_attribute_spec(
            "NAME_PORTFOLIO",
            "Was the previous application for",  # TODO: réviser name_metier
            "name_portfolio",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Cards": "cards",
                "Cars": "cars",
                "Cash": "cash",
                "POS": "pos",
                "XNA": "xna",
            },
        ),  # card=5

        build_attribute_spec(
            "NAME_PRODUCT_TYPE",
            "Was the previous application x-sell",  # TODO: réviser name_metier
            "name_product_type",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "XNA": "xna",
                "walk-in": "walk_in",
                "x-sell": "x_sell",
            },
        ),  # card=3

        build_attribute_spec(
            "CHANNEL_TYPE",
            "Through which channel we acquired",  # TODO: réviser name_metier
            "channel_type",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "AP+ (Cash loan)": "ap_cash_loan",
                "Car dealer": "car_dealer",
                "Channel of corporate sales": "channel_of_corporate_sales",
                "Contact center": "contact_center",
                "Country-wide": "country_wide",
                "Credit and cash offices": "credit_and_cash_offices",
                "Regional / Local": "regional_local",
                "Stone": "stone",
            },
        ),  # card=8

        build_attribute_spec(
            "SELLERPLACE_AREA",
            "Selling area of seller place",  # TODO: réviser name_metier
            "sellerplace_area",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-1.0, max=4000000.0

        build_attribute_spec(
            "NAME_SELLER_INDUSTRY",
            "The industry of the seller",  # TODO: réviser name_metier
            "name_seller_industry",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Auto technology": "auto_technology",
                "Clothing": "clothing",
                "Connectivity": "connectivity",
                "Construction": "construction",
                "Consumer electronics": "consumer_electronics",
                "Furniture": "furniture",
                "Industry": "industry",
                "Jewelry": "jewelry",
                "MLM partners": "mlm_partners",
                "Tourism": "tourism",
                "XNA": "xna",
            },
        ),  # card=11

        build_attribute_spec(
            "CNT_PAYMENT",
            "Term of previous credit at",  # TODO: réviser name_metier
            "cnt_payment",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=60.0

        build_attribute_spec(
            "NAME_YIELD_GROUP",
            "Grouped interest rate into small",  # TODO: réviser name_metier
            "name_yield_group",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "XNA": "xna",
                "high": "high",
                "low_action": "low_action",
                "low_normal": "low_normal",
                "middle": "middle",
            },
        ),  # card=5

        build_attribute_spec(
            "PRODUCT_COMBINATION",
            "Detailed product combination of the",  # TODO: réviser name_metier
            "product_combination",
            source_table="previous_application",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ORDINAL,
        ),  # card=17

        build_attribute_spec(
            "DAYS_FIRST_DRAWING",
            "Relative to application date of",  # TODO: réviser name_metier
            "days_first_drawing",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-2910.0, max=365243.0 | ⚠️ 38% null

        build_attribute_spec(
            "DAYS_FIRST_DUE",
            "Relative to application date of",  # TODO: réviser name_metier
            "days_first_due",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-2891.0, max=365243.0 | ⚠️ 38% null

        build_attribute_spec(
            "DAYS_LAST_DUE_1ST_VERSION",
            "Relative to application date of",  # TODO: réviser name_metier
            "days_last_due_1st_version",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-2800.0, max=365243.0 | ⚠️ 38% null

        build_attribute_spec(
            "DAYS_LAST_DUE",
            "Relative to application date of",  # TODO: réviser name_metier
            "days_last_due",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-2850.0, max=365243.0 | ⚠️ 38% null

        build_attribute_spec(
            "DAYS_TERMINATION",
            "Relative to application date of",  # TODO: réviser name_metier
            "days_termination",
            source_table="previous_application",
            transform=TransformType.STANDARD,
        ),  # min=-2844.0, max=365243.0 | ⚠️ 38% null

        build_attribute_spec(
            "NFLAG_INSURED_ON_APPROVAL",
            "Did the client requested insurance",  # TODO: réviser name_metier
            "nflag_insured_on_approval",
            source_table="previous_application",
            col_type=ColumnType.BINARY,
            valeurs_possibles={
                "0.0": "v_0_0",
                "1.0": "v_1_0",
            },
        ),  # card=2 | ⚠️ 38% null


        # ════════════════════════════════════════════════════════════
        # POS CASH
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_PREV",
            "ID of previous credit in",  # TODO: réviser name_metier
            "sk_id_prev",
            source_table="pos_cash",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "SK_ID_CURR",
            "ID of loan in our",  # TODO: réviser name_metier
            "sk_id_curr",
            source_table="pos_cash",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "MONTHS_BALANCE",
            "Month of balance relative to",  # TODO: réviser name_metier
            "months_balance",
            source_table="pos_cash",
            transform=TransformType.STANDARD,
        ),  # min=-96.0, max=-1.0

        build_attribute_spec(
            "CNT_INSTALMENT",
            "Term of previous credit",  # TODO: réviser name_metier
            "cnt_instalment",
            source_table="pos_cash",
            transform=TransformType.STANDARD,
        ),  # min=1.0, max=66.0

        build_attribute_spec(
            "CNT_INSTALMENT_FUTURE",
            "Installments left to pay on",  # TODO: réviser name_metier
            "cnt_instalment_future",
            source_table="pos_cash",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=61.0

        build_attribute_spec(
            "NAME_CONTRACT_STATUS",
            "Contract status of previous application",  # TODO: réviser name_metier
            "name_contract_status",
            source_table="pos_cash",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Active": "active",
                "Approved": "approved",
                "Completed": "completed",
                "Demand": "demand",
                "Returned to the store": "returned_to_the_store",
                "Signed": "signed",
            },
        ),  # card=6

        build_attribute_spec(
            "SK_DPD",
            "DPD during the month on",  # TODO: réviser name_metier
            "sk_dpd",
            source_table="pos_cash",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=2214.0

        build_attribute_spec(
            "SK_DPD_DEF",
            "DPD during the month with",  # TODO: réviser name_metier
            "sk_dpd_def",
            source_table="pos_cash",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=203.0


        # ════════════════════════════════════════════════════════════
        # CREDIT CARD
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_PREV",
            "ID of previous credit in",  # TODO: réviser name_metier
            "sk_id_prev",
            source_table="credit_card",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "SK_ID_CURR",
            "ID of loan in our",  # TODO: réviser name_metier
            "sk_id_curr",
            source_table="credit_card",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "MONTHS_BALANCE",
            "Month of balance relative to",  # TODO: réviser name_metier
            "months_balance",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=-96.0, max=-1.0

        build_attribute_spec(
            "AMT_BALANCE",
            "Balance during the month of",  # TODO: réviser name_metier
            "amt_balance",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=-135359.01, max=959924.47

        build_attribute_spec(
            "AMT_CREDIT_LIMIT_ACTUAL",
            "Credit card limit during the",  # TODO: réviser name_metier
            "amt_credit_limit_actual",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=1350000.0

        build_attribute_spec(
            "AMT_DRAWINGS_ATM_CURRENT",
            "Amount drawing at ATM during",  # TODO: réviser name_metier
            "amt_drawings_atm_current",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=1305000.0

        build_attribute_spec(
            "AMT_DRAWINGS_CURRENT",
            "Amount drawing during the month",  # TODO: réviser name_metier
            "amt_drawings_current",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=1305000.0

        build_attribute_spec(
            "AMT_DRAWINGS_OTHER_CURRENT",
            "Amount of other drawings during",  # TODO: réviser name_metier
            "amt_drawings_other_current",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=761940.0

        build_attribute_spec(
            "AMT_DRAWINGS_POS_CURRENT",
            "Amount drawing or buying goods",  # TODO: réviser name_metier
            "amt_drawings_pos_current",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=900000.0

        build_attribute_spec(
            "AMT_INST_MIN_REGULARITY",
            "Minimal installment for this month",  # TODO: réviser name_metier
            "amt_inst_min_regularity",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=46728.54

        build_attribute_spec(
            "AMT_PAYMENT_CURRENT",
            "How much did the client",  # TODO: réviser name_metier
            "amt_payment_current",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=916812.0

        build_attribute_spec(
            "AMT_PAYMENT_TOTAL_CURRENT",
            "How much did the client",  # TODO: réviser name_metier
            "amt_payment_total_current",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=0.0, max=900000.0

        build_attribute_spec(
            "AMT_RECEIVABLE_PRINCIPAL",
            "Amount receivable for principal on",  # TODO: réviser name_metier
            "amt_receivable_principal",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=-135867.11, max=899968.59

        build_attribute_spec(
            "AMT_RECIVABLE",
            "Amount receivable on the previous",  # TODO: réviser name_metier
            "amt_recivable",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=-133442.23, max=957233.47

        build_attribute_spec(
            "AMT_TOTAL_RECEIVABLE",
            "Total amount receivable on the",  # TODO: réviser name_metier
            "amt_total_receivable",
            source_table="credit_card",
            transform=TransformType.LOG,
        ),  # min=-133442.23, max=957233.47

        build_attribute_spec(
            "CNT_DRAWINGS_ATM_CURRENT",
            "Number of drawings at ATM",  # TODO: réviser name_metier
            "cnt_drawings_atm_current",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=33.0

        build_attribute_spec(
            "CNT_DRAWINGS_CURRENT",
            "Number of drawings during this",  # TODO: réviser name_metier
            "cnt_drawings_current",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=101.0

        build_attribute_spec(
            "CNT_DRAWINGS_OTHER_CURRENT",
            "Number of other drawings during",  # TODO: réviser name_metier
            "cnt_drawings_other_current",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=6.0

        build_attribute_spec(
            "CNT_DRAWINGS_POS_CURRENT",
            "Number of drawings for goods",  # TODO: réviser name_metier
            "cnt_drawings_pos_current",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=101.0

        build_attribute_spec(
            "CNT_INSTALMENT_MATURE_CUM",
            "Number of paid installments on",  # TODO: réviser name_metier
            "cnt_instalment_mature_cum",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=116.0

        build_attribute_spec(
            "NAME_CONTRACT_STATUS",
            "Contract status of previous application",  # TODO: réviser name_metier
            "name_contract_status",
            source_table="credit_card",
            col_type=ColumnType.CATEGORICAL,
            encoding=EncodingType.ONE_HOT,
            valeurs_possibles={
                "Active": "active",
                "Completed": "completed",
                "Demand": "demand",
                "Sent proposal": "sent_proposal",
                "Signed": "signed",
            },
        ),  # card=5

        build_attribute_spec(
            "SK_DPD",
            "DPD during the month on",  # TODO: réviser name_metier
            "sk_dpd",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=2284.0

        build_attribute_spec(
            "SK_DPD_DEF",
            "DPD during the month with",  # TODO: réviser name_metier
            "sk_dpd_def",
            source_table="credit_card",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=1217.0


        # ════════════════════════════════════════════════════════════
        # INSTALLMENTS
        # ════════════════════════════════════════════════════════════

        build_attribute_spec(
            "SK_ID_PREV",
            "ID of previous credit in",  # TODO: réviser name_metier
            "sk_id_prev",
            source_table="installments",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "SK_ID_CURR",
            "ID of loan in our",  # TODO: réviser name_metier
            "sk_id_curr",
            source_table="installments",
            role=ColumnRole.IDENTIFIER,
            col_type=ColumnType.IDENTIFIER,
        ),

        build_attribute_spec(
            "NUM_INSTALMENT_VERSION",
            "Version of installment calendar of",  # TODO: réviser name_metier
            "num_instalment_version",
            source_table="installments",
            transform=TransformType.STANDARD,
        ),  # min=0.0, max=34.0

        build_attribute_spec(
            "NUM_INSTALMENT_NUMBER",
            "On which installment we observe",  # TODO: réviser name_metier
            "num_instalment_number",
            source_table="installments",
            transform=TransformType.STANDARD,
        ),  # min=1.0, max=216.0

        build_attribute_spec(
            "DAYS_INSTALMENT",
            "When the installment of previous",  # TODO: réviser name_metier
            "days_instalment",
            source_table="installments",
            transform=TransformType.STANDARD,
        ),  # min=-2922.0, max=-2.0

        build_attribute_spec(
            "DAYS_ENTRY_PAYMENT",
            "When was the installments of",  # TODO: réviser name_metier
            "days_entry_payment",
            source_table="installments",
            transform=TransformType.STANDARD,
        ),  # min=-2989.0, max=-2.0

        build_attribute_spec(
            "AMT_INSTALMENT",
            "What was the prescribed installment",  # TODO: réviser name_metier
            "amt_instalment",
            source_table="installments",
            transform=TransformType.LOG,
        ),  # min=0.0, max=2292736.86

        build_attribute_spec(
            "AMT_PAYMENT",
            "What the client actually paid",  # TODO: réviser name_metier
            "amt_payment",
            source_table="installments",
            transform=TransformType.LOG,
        ),  # min=0.0, max=2292736.86










    # #########################################################################
    # v_agg_bureau (13)
    # #########################################################################        
    # bureau_credit_count
    # bureau_amt_credit_sum_mean
    # bureau_amt_credit_sum_total
    # bureau_days_credit_max
    # bureau_days_credit_min
    # bureau_amt_credit_sum_debt_mean
    # bureau_amt_credit_sum_debt_total
    # bureau_active_credit_count
    # bureau_closed_credit_count
    # bureau_overdue_count
    # bureau_credit_day_overdue_mean
    # bureau_cnt_prolong_mean
    # bureau_credit_limit_mean
    # #########################################################################
        
        build_attribute_spec(
            "BUREAU_CREDIT_COUNT",
            "Nb crédits bureau",
            "bureau_credit_count",
            source_table="bureau_agg",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "BUREAU_AMT_CREDIT_SUM_MEAN",
            "Montant moy bureau",
            "bureau_amt_credit_sum_mean",
            source_table="bureau_agg",
            transform=TransformType.LOG,
        ),
        build_attribute_spec(
            "BUREAU_DAYS_CREDIT_MAX",
            "Crédit récent",
            "bureau_days_credit_max",
            source_table="bureau_agg",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "BUREAU_ACTIVE_CREDIT_COUNT",
            "Crédits actifs",
            "bureau_active_credit_count",
            source_table="bureau_agg",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN",
            "Dette moy. bureau",
            "bureau_amt_credit_sum_debt_mean",
            source_table="bureau_agg",
            transform=TransformType.LOG,
        ),
        build_attribute_spec(
            "BUREAU_OVERDUE_COUNT",
            "Nb retards bureau",
            "bureau_overdue_count",
            source_table="bureau_agg",
            transform=TransformType.STANDARD,
        ),

    # ── bureau_agg (columnas faltantes) ──────────────────────────────────────
        build_attribute_spec(
            "bureau_amt_credit_sum_total",
            "Total montant crédits bureau",
            "bureau_amt_credit_sum_total",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ),
        build_attribute_spec(
            "bureau_days_credit_min",
            "Ancienneté min crédit bureau",
            "bureau_days_credit_min",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ),
        build_attribute_spec(
            "bureau_amt_credit_sum_debt_total",
            "Total dette bureau",
            "bureau_amt_credit_sum_debt_total",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ),
        build_attribute_spec(
            "bureau_closed_credit_count",
            "Nb crédits bureau clôturés",
            "bureau_closed_credit_count",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ),
        build_attribute_spec(
            "bureau_credit_day_overdue_mean",
            "Nb jours retard moyen bureau",
            "bureau_credit_day_overdue_mean",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ),
        build_attribute_spec(
            "bureau_cnt_prolong_mean",
            "Nb prolongations moyen bureau",
            "bureau_cnt_prolong_mean",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ),
        build_attribute_spec(
            "bureau_credit_limit_mean",
            "Limite crédit moyen bureau",
            "bureau_credit_limit_mean",
            source_table="bureau_agg",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ),

    # #########################################################################
    # v_agg_previous (11)
    # #########################################################################
    # prev_app_count
    # prev_approved_count
    # prev_refused_count
    # prev_canceled_count
    # prev_amt_credit_mean
    # prev_amt_credit_max
    # prev_amt_annuity_mean
    # prev_days_decision_max
    # prev_rate_down_mean
    # prev_cnt_payment_mean
    # prev_refusal_rate
    # #########################################################################
        
        build_attribute_spec(
            "PREV_APP_COUNT",
            "Nb demandes",
            "prev_app_count",
            source_table="v_agg_previous",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "PREV_APPROVED_COUNT",
            "Nb approuvées",
            "prev_approved_count",
            source_table="v_agg_previous",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "PREV_REFUSED_COUNT",
            "Nb refusées",
            "prev_refused_count",
            source_table="v_agg_previous",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "PREV_AMT_CREDIT_MEAN",
            "Montant précéd.",
            "prev_amt_credit_mean",
            source_table="v_agg_previous",
            transform=TransformType.LOG,
        ),
        build_attribute_spec(
            "PREV_DAYS_DECISION_MAX",
            "Décision récente",
            "prev_days_decision_max",
            source_table="v_agg_previous",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "PREV_CANCELED_COUNT",
            "Prev Canceled Count",
            "prev_canceled_count",
            source_table="v_agg_previous",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=1 | 1.8% null
        build_attribute_spec(
            "PREV_AMT_CREDIT_MAX",
            "Prev Amt Credit Max",
            "prev_amt_credit_max",
            source_table="v_agg_previous",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=3027 | 1.8% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "PREV_AMT_ANNUITY_MEAN",
            "Prev Amt Annuity Mean",
            "prev_amt_annuity_mean",
            source_table="v_agg_previous",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=4889 | 1.9% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "PREV_RATE_DOWN_MEAN",
            "Prev Rate Down Mean",
            "prev_rate_down_mean",
            source_table="v_agg_previous",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=2987 | 7.7% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "PREV_CNT_PAYMENT_MEAN",
            "Prev Cnt Payment Mean",
            "prev_cnt_payment_mean",
            source_table="v_agg_previous",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=543 | 1.9% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "PREV_REFUSAL_RATE",
            "Prev Refusal Rate",
            "prev_refusal_rate",
            source_table="v_agg_previous",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=1 | 1.8% null
        
    #########################################################################
    # v_agg_installments (9)
    #########################################################################
    # install_records_count
    # install_payment_delay_mean
    # install_dpd_max
    # install_late_count
    # install_payment_ratio_mean
    # install_payment_ratio_min
    # install_amt_payment_mean
    # install_amt_instalment_mean
    # install_late_rate
    #########################################################################
        
        build_attribute_spec(
            "INSTALL_PAYMENT_DELAY_MEAN",
            "Retard paiement",
            "install_payment_delay_mean",
            source_table="v_agg_installments",
            transform=TransformType.ROBUST,
        ),
        build_attribute_spec(
            "INSTALL_PAYMENT_RATIO_MEAN",
            "Ratio paiement",
            "install_payment_ratio_mean",
            source_table="v_agg_installments",
            transform=TransformType.STANDARD,
        ),
        build_attribute_spec(
            "INSTALL_DPD_MAX",
            "Retard max",
            "install_dpd_max",
            source_table="v_agg_installments",
            transform=TransformType.ROBUST,
        ),

        build_attribute_spec(
            "INSTALL_RECORDS_COUNT",
            "Install Records Count",
            "install_records_count",
            source_table="v_agg_installments",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=229 | 1.6% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "INSTALL_LATE_COUNT",
            "Install Late Count",
            "install_late_count",
            source_table="v_agg_installments",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=56 | 1.6% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "INSTALL_PAYMENT_RATIO_MIN",
            "Install Payment Ratio Min",
            "install_payment_ratio_min",
            source_table="v_agg_installments",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=2020 | 1.6% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "INSTALL_AMT_PAYMENT_MEAN",
            "Install Amt Payment Mean",
            "install_amt_payment_mean",
            source_table="v_agg_installments",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=4922 | 1.6% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "INSTALL_AMT_INSTALMENT_MEAN",
            "Install Amt Instalment Mean",
            "install_amt_instalment_mean",
            source_table="v_agg_installments",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=4922 | 1.6% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "INSTALL_LATE_RATE",
            "Install Late Rate",
            "install_late_rate",
            source_table="v_agg_installments",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=853 | 1.6% null | ⚠️ HAUTE CARDINALITÉ
    #########################################################################
    # v_agg_pos_cash (8)
    #########################################################################
    # pos_records_count
    # pos_months_balance_mean
    # pos_sk_dpd_max
    # pos_sk_dpd_mean
    # pos_sk_dpd_def_max
    # pos_cnt_instalment_mean
    # pos_cnt_instalment_future_mean   # TODO !!!!
    # pos_dpd_count
    #########################################################################
        
        build_attribute_spec(
            "POS_RECORDS_COUNT",
            "Pos Records Count",
            "pos_records_count",
            source_table="v_agg_pos_cash",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=150 | 1.8% null | ⚠️ HAUTE CARDINALITÉ
        
        build_attribute_spec(
            "POS_MONTHS_BALANCE_MEAN",
            "Solde POS moy.",
            "pos_months_balance_mean",
            source_table="v_agg_pos_cash",
            transform=TransformType.STANDARD,
        ),
        
        build_attribute_spec(
            "POS_SK_DPD_MAX",
            "Retard max POS",
            "pos_sk_dpd_max",
            source_table="v_agg_pos_cash",
            transform=TransformType.ROBUST,
        ),

        build_attribute_spec(
            "POS_SK_DPD_MEAN",
            "Pos Sk Dpd Mean",
            "pos_sk_dpd_mean",
            source_table="v_agg_pos_cash",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=604 | 1.8% null | ⚠️ HAUTE CARDINALITÉ
        
        build_attribute_spec(
            "POS_SK_DPD_DEF_MAX",
            "Pos Sk Dpd Def Max",
            "pos_sk_dpd_def_max",
            source_table="v_agg_pos_cash",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=29 | 1.8% null
        
        build_attribute_spec(
            "POS_CNT_INSTALMENT_MEAN",
            "Pos Cnt Instalment Mean",
            "pos_cnt_instalment_mean",
            source_table="v_agg_pos_cash",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=2595 | 1.8% null | ⚠️ HAUTE CARDINALITÉ
        
        build_attribute_spec(
            "POS_DPD_COUNT",
            "Pos Dpd Count",
            "pos_dpd_count",
            source_table="v_agg_pos_cash",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=36 | 1.8% null
    
    #########################################################################
    # v_agg_credit_card (11)
    #########################################################################
    # cc_records_count
    # cc_amt_balance_mean
    # cc_amt_balance_max
    # cc_credit_limit_mean
    # cc_amt_drawings_current_sum
    # cc_amt_drawings_current_mean
    # cc_payment_total_mean
    # cc_sk_dpd_max
    # cc_sk_dpd_mean
    # cc_cnt_drawings_mean    # TODO 
    # cc_dpd_count
    #########################################################################
        
        build_attribute_spec(
            "CC_AMT_BALANCE_MEAN",
            "Solde CC moy.",
            "cc_amt_balance_mean",
            source_table="v_agg_credit_card",
            transform=TransformType.LOG,
        ),
        build_attribute_spec(
            "CC_SK_DPD_MAX",
            "Retard max CC",
            "cc_sk_dpd_max",
            source_table="v_agg_credit_card",
            transform=TransformType.ROBUST,
        ),
        build_attribute_spec(
            "CC_AMT_DRAWINGS_CURRENT_SUM",
            "Total retraits CC",
            "cc_amt_drawings_current_sum",
            source_table="v_agg_credit_card",
            transform=TransformType.LOG,
        ),

        build_attribute_spec(
            "CC_RECORDS_COUNT",
            "Cc Records Count",
            "cc_records_count",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=98 | 66.2% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "CC_AMT_BALANCE_MAX",
            "Cc Amt Balance Max",
            "cc_amt_balance_max",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=1046 | 66.2% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "CC_CREDIT_LIMIT_MEAN",
            "Cc Credit Limit Mean",
            "cc_credit_limit_mean",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=759 | 66.2% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "CC_AMT_DRAWINGS_CURRENT_MEAN",
            "Cc Amt Drawings Current Mean",
            "cc_amt_drawings_current_mean",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=1044 | 66.2% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "CC_PAYMENT_TOTAL_MEAN",
            "Cc Payment Total Mean",
            "cc_payment_total_mean",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=1084 | 66.2% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "CC_SK_DPD_MEAN",
            "Cc Sk Dpd Mean",
            "cc_sk_dpd_mean",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=181 | 66.2% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "CC_DPD_COUNT",
            "Cc Dpd Count",
            "cc_dpd_count",
            source_table="v_agg_credit_card",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=36 | 66.2% null
        
    #########################################################################
    # engineered (10)
    #########################################################################

        build_attribute_spec(
            "FE1_CREDIT_INCOME_RATIO",
            "Fe1 Credit Income Ratio",
            "fe1_credit_income_ratio",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=2886 | 0.0% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE2_ANNUITY_INCOME_RATIO",
            "Fe2 Annuity Income Ratio",
            "fe2_annuity_income_ratio",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=4339 | 0.1% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE3_PAYMENT_RATE",
            "Fe3 Payment Rate",
            "fe3_payment_rate",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=2636 | 0.1% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE4_DAYS_EMPLOYED_RATIO",
            "Fe4 Days Employed Ratio",
            "fe4_days_employed_ratio",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=4027 | 19.5% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE5_AGE_YEARS",
            "Fe5 Age Years",
            "fe5_age_years",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=4295 | 0.0% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE6_INCOME_PER_PERSON",
            "Fe6 Income Per Person",
            "fe6_income_per_person",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.LOG,
        ), # card=291 | 0.0% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE7_EXT_SOURCES_MEAN",
            "Fe7 Ext Sources Mean",
            "fe7_ext_sources_mean",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=4995 | 0.0% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE8_EXT_SOURCES_MIN",
            "Fe8 Ext Sources Min",
            "fe8_ext_sources_min",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=3645 | 0.0% null | ⚠️ HAUTE CARDINALITÉ
        build_attribute_spec(
            "FE9_BUREAU_OVERDUE_RATE",
            "Fe9 Bureau Overdue Rate",
            "fe9_bureau_overdue_rate",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=21 | 0.0% null
        build_attribute_spec(
            "FE10_COMPOSITE_RISK_SCORE",
            "Fe10 Composite Risk Score",
            "fe10_composite_risk_score",
            source_table="engineered",
            col_type=ColumnType.NUMERICAL,
            transform=TransformType.NONE,
        ), # card=4987 | 0.0% null | ⚠️ HAUTE CARDINALITÉ
        
    ]
)





# En bas de schema.py, temporairement :
# from src.data.schema_bootstrap import REGISTRY_BOOTSTRAP

# Vérifier que tout est cohérent

#print("REGISTRY attributes counter:")    
#print(len(REGISTRY.attributes))   # comparer avec REGISTRY actuel
#print(REGISTRY.cols_ohe)
#print(REGISTRY.cols_log)

#print("REGISTRY")    

#print(len(REGISTRY_2.attributes))   # comparer avec REGISTRY actuel
#print(REGISTRY_2.cols_ohe)
#print(REGISTRY_2.cols_log)

# Quand c'est bon → remplacer REGISTRY par REGISTRY_BOOTSTRAP et renommer
