"""
src/data/schema_bootstrap.py
================================
FICHIER AUTO-GÉNÉRÉ par bootstrap_registry.py
NE PAS UTILISER DIRECTEMENT — réviser puis copier dans schema.py

Workflow :
  1. Réviser tous les "TODO: réviser name_metier" → nom métier humain court
  2. Vérifier les valeurs_possibles générées automatiquement
  3. Ajuster ColumnType / EncodingType si nécessaire
  4. Copier le bloc REGISTRY dans schema.py

Légende des commentaires inline :
  ⚠️ X% null  → taux de valeurs manquantes élevé
  card=N       → cardinalité (nb valeurs uniques)
  min=X, max=Y → plage de valeurs
"""

from __future__ import annotations
from typing import Optional, Dict, List
from dataclasses import dataclass, field

# --- Importer depuis schema.py ---
from src.data.schema import (
    AttributeSpec, FeatureRegistry, ColumnType, ColumnRole,
    EncodingType, TransformType, build_attribute_spec
)


# ═══════════════════════════════════════════════════════════════════════
# REGISTRY BOOTSTRAP — généré automatiquement, à réviser avant usage
# ═══════════════════════════════════════════════════════════════════════

REGISTRY_BOOTSTRAP = FeatureRegistry(
    attributes=[

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
            encoding=EncodingType.TARGET_ENCODING,
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

    ]
)


# Pour tester le registry généré :
if __name__ == "__main__":
    print(f"Total attributs : {len(REGISTRY_BOOTSTRAP.attributes)}")
    print(f"Cols OHE        : {REGISTRY_BOOTSTRAP.cols_ohe}")
    print(f"Cols LOG        : {REGISTRY_BOOTSTRAP.cols_log}")
    print(f"Cols STANDARD   : {REGISTRY_BOOTSTRAP.cols_standard[:5]}...")
