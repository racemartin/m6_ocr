
### 📊 Estrategia de Búsqueda de Hiperparámetros (Grid vs. Optuna)

| Modèle | Hyperparamètre | Explication (Didactique) | Recherche Phase 3 (Grille) | Espace Optuna (Phase 4) |
| --- | --- | --- | --- | --- |
| **Dummy** | `strategy` | Méthode de prédiction aléatoire pour la baseline. | `stratified`, `most_frequent` | - |
| **Logistic Regression** | `C` | Inverse de la force de régularisation (plus petit = plus de régularisation). | `0.01, 0.1, 1.0, 10.0` | `1e-4` à `10.0` (Log-scale) |
|  | `solver` | Algorithme d'optimisation numérique. | `lbfgs`, `liblinear` | Catégoriel : `lbfgs`, `liblinear` |
| **Decision Tree** | `max_depth` | Profondeur maximale de l'arbre (contrôle l'overfitting). | `3, 5, 8, 12` | Entier : `3` à `15` |
|  | `min_samples_leaf` | Nombre minimum d'échantillons requis dans une feuille. | `20, 50, 100` | Entier : `10` à `200` |
| **Random Forest** | `n_estimators` | Nombre d'arbres dans la forêt (bagging). | `100, 200` | Entier : `100` à `500` |
|  | `max_features` | Nb de features considérées pour chaque split. | `sqrt`, `log2` | Catégoriel : `sqrt`, `log2` |
| **Gradient Boosting** | `learning_rate` | Poids accordé à chaque nouvel arbre (vitesse d'apprentissage). | `0.03, 0.05, 0.10` | `0.01` à `0.3` (Log-scale) |
|  | `subsample` | Fraction des données utilisées pour entraîner chaque arbre. | - | `0.6` à `1.0` (Continu) |
| **XGBoost** | `scale_pos_weight` | **Crucial :** Poids de la classe minoritaire (Target=1). | `11, 12` | `5.0` à `20.0` (Ajustement fin) |
|  | `reg_alpha` / `lambda` | Régularisations L1 et L2 pour éviter la mémorisation. | - | `1e-8` à `10.0` (Log-scale) |
|  | `colsample_bytree` | Fraction de colonnes échantillonnées par arbre. | `0.7, 0.8` | `0.5` à `1.0` (Continu) |
| **LightGBM** | `num_leaves` | Nombre max de feuilles (plus précis que la profondeur seule). | `31, 50, 80` | Entier : `20` à `150` |
| **MLP (Réseau)** | `hidden_layer_sizes` | Architecture du réseau (nombre de neurones/couches). | `(128,64)`, `(64,32)` | Dynamique : `1` à `3` couches |
|  | `smote__sampling` | Taux de synthèse de données (Phase 3 uniquement ici). | `0.25, 0.40` | - (Inclus dans l'architecture) |

---

### 💡 Notas 

1. **Log-Scale **: En Optuna usamos `log=True` para parámetros como `C` o `learning_rate` porque un cambio de `0.001` a `0.01` es tan importante como uno de `0.1` a `1.0`. Es una búsqueda por órdenes de magnitud.
2. **Arquitectura Dinámica (Geoffrey Hinton)**: Nota que en tu código de Optuna para el **MLP**, has implementado algo avanzado: Optuna no solo elige el número de neuronas, sino también el **número de capas** (`n_layers`) dinámicamente. Esto es casi "AutoML".
3. **Déséquilibre (Yann LeCun)**: En **XGBoost**, el parámetro `scale_pos_weight` es nuestra mejor arma. En la Fase 3 probamos valores cercanos al ratio real (11.5), pero en la Fase 4 dejamos que Optuna encuentre el punto exacto que maximiza el **F2-Score**.





HACEMOS ...
    # --------------------------------------------------------------------------
    # STEP 9 : ENREGISTREMENT DES BENCHMARKS DE MONITORING
    # --------------------------------------------------------------------------
    def step9_register_monitoring(self):
        """
        Insère les métriques de référence dans 'monitoring_metrics'.
        Indispensable pour la Phase 6 (Dashboard) afin de comparer le 
        réel vs le théorique.
        """
        

### 1. Detección de Model Decay (Degradación)

En el mundo real, los datos cambian (*Data Drift*). Por ejemplo, si hay una crisis económica, el perfil de los clientes de Home Credit cambiará respecto a 2018.

* **Utilidad**: Tu Dashboard comparará el `f2_score` que guardaste hoy (ej. **0.28**) con el rendimiento de las predicciones reales de mañana.
* **Acción**: Si el score real cae por debajo del **20% de la referencia** guardada en la Fase 4, el sistema te avisará: *"¡Cuidado! El modelo ya no es fiable, es hora de re-entrenar"*.

### 2. Contraste en el Dashboard de Negocio (Fase 6)

Cuando presentes tu proyecto, el usuario (un analista de riesgos) verá el Dashboard en **Streamlit**.

* **Utilidad**: Podrás mostrar una tarjeta que diga: **"Rendimiento Teórico (Benchmark): 0.28"** al lado de **"Rendimiento Actual: 0.27"**.
* **Valor**: Esto genera confianza. El banco sabe qué esperar del modelo porque tiene una "hoja de especificaciones" guardada en la base de datos.

### 3. Ajuste Dinámico del Umbral (Optimal Threshold)

Guardar el `ref_threshold` es vital para la **Fase 5 (API)**.

* **Utilidad**: La API de predicción no usará `0.5` para decidir si otorga un crédito, sino que consultará esta tabla para saber que el umbral óptimo es, por ejemplo, `0.32`.
* **Acción**: Si en el futuro optimizas el modelo otra vez y el umbral cambia, la API se actualizará automáticamente simplemente leyendo el último valor de esta tabla, **sin que tengas que tocar una sola línea de código del servidor**.

### 4. Auditoría y Gobernanza (Nivel Universitario)

En banca, si se deniega un crédito, a veces hay que explicar por qué y bajo qué criterios se tomó la decisión.

* **Utilidad**: Al tener vinculadas las métricas con el `model_version_id`, tienes una trazabilidad completa. Puedes demostrar ante una auditoría que el modelo `v_20260223` fue validado con un F2-Score específico y bajo qué condiciones de riesgo.

---

### Resumen de utilidad por tabla:

| Dato Guardado | ¿Para qué sirve luego? |
| --- | --- |
| **`ref_f2_score`** | Alerta de degradación (Si baja el real, re-entrenar). |
| **`ref_threshold`** | Configura la lógica de decisión de la API (Fase 5). |
| **`ref_recall`** | Asegura que el modelo sigue detectando a los morosos (FN). |
| **`metadata`** | Diferencia si la métrica es de Laboratorio o de Producción. |



<img src="docs\images\Flux_de_Decision_Phase_3vsPhase_4.png" alt="Phase_4_Optimisation_Bayesienne" width="900">

```
@startuml
skinparam backgroundColor #FFFFFF
skinparam ActivityBackgroundColor #F9F9F9

title Flux de Décision : Phase 3 vs Phase 4

start

partition "Phase 3 : Sélection du Champion" {
    :Exécuter GridSearchCV sur tous les modèles;
    note right: Logistic, RF, XGB, LGBM, MLP
    :Comparer les F2-Scores (Cross-Val);
    :Identifier le **Champion Initial**;
    :Enregistrer les résultats dans MLflow;
}

partition "Phase 4 : Optimisation Fine" {
    :Charger le Champion de la Phase 3;
    
    fork
        :Maintenir les paramètres GridSearch;
    fork again
        :Lancer l'étude Optuna (TPE);
        note right: Recherche bayésienne continue
    end merge

    :Comparer **GridSearch Best** vs **Optuna Best**;
    
    if (Optuna a amélioré le F2-Score?) then (Oui)
        :Mettre à jour le modèle avec Optuna Params;
    else (Non)
        :Garder les paramètres de la Phase 3;
    endif
}

partition "Finalisation" {
    :Optimisation du Seuil (Threshold);
    :Exportation finale (PostgreSQL / MLflow);
}

stop
@enduml

```

<img src="docs\images\Hyperparameter_Tuning_Business_Threshold.png" alt="Phase_4_Optimisation_Bayesienne" width="900">




```
@startuml
skinparam style strictuml
skinparam sequenceMessageAlign center
skinparam BoxPadding 10

title Diagrama de Secuencia: Phase 4 — Hyperparameter Tuning & Business Threshold

actor "Data Scientist" as User
participant "Phase4Pipeline" as P4
database "File System\n(models/ & data/)" as FS
database "PostgreSQL\n(DB)" as DB
entity "MLflow Tracking\nServer" as MLF

== INIT & SETUP ==

User -> P4 : __init__(champion_name, engine, ...)
activate P4
P4 -> P4 : Initialize attributes (Paths, Scores, etc.)
User -> P4 : step0_setup()
P4 -> MLF : set_tracking_uri()
P4 -> MLF : get_experiment_by_name()
alt Experiment exists
    MLF --> P4 : experiment_id
else Experiment doesn't exist
    P4 -> MLF : create_experiment()
    MLF --> P4 : new experiment_id
end
P4 --> User : Success: MLflow configured

== STEP 1: LOAD CHAMPION ==

User -> P4 : step1_load_champion()
alt champion_name is None
    P4 -> P4 : _detect_champion()
    P4 -> FS : glob models/*_metadata.json
    FS --> P4 : metadata files
end
P4 -> FS : load(joblib model_path)
P4 -> FS : load(json meta_path)
FS --> P4 : Champion Model & Metadata
P4 --> User : Success: Champion loaded

== STEP 2: LOAD & SPLIT DATA ==

User -> P4 : step2_load_data()
alt source == "db"
    P4 -> DB : SELECT * FROM v_features_engineering
    DB --> P4 : df_raw
else source == "csv"
    P4 -> FS : read_csv(X_train, y_train)
    FS --> P4 : df_raw
end
P4 -> P4 : _split_data(eval_ratio, random_state)
note right: Reproduces same split as Phase 3\nto avoid data leakage
P4 --> User : Success: X_train, X_eval ready

== STEP 3 & 4: OPTIMIZATION (Simplified) ==

group Loop per Engine (GridSearch / Optuna)
    User -> P4 : step3_gridsearch() / step4_optuna()
    P4 -> P4 : Cross-Validation (cv_folds)
    P4 -> MLF : log_params(), log_metrics()
    P4 -> MLF : log_model()
    MLF --> P4 : run_id
end

== STEP 6, 7 & 8: THRESHOLD & PERSISTENCE ==

User -> P4 : step6_optimize_threshold()
P4 -> P4 : Cost Scan [0.05, 0.80]
note right: Cost = FN*5 + FP*1
P4 --> User : optimal_threshold found

User -> P4 : step7_save()
P4 -> FS : Save best_model.joblib
P4 -> FS : Save metadata.json (with eval_metrics)
P4 --> User : Success: Artifacts saved

User -> P4 : step8_register_db()
P4 -> DB : INSERT INTO model_versions (RETURNING id)
DB --> P4 : model_version_id
P4 -> P4 : step9_register_monitoring()
P4 -> DB : INSERT INTO monitoring_metrics
P4 --> User : Success: Pipeline Phase 4 Finished

deactivate P4
@enduml
```



<img src="docs\images\Phase_4_Optimisation_Bayesienne.png" alt="Phase_4_Optimisation_Bayesienne" width="900">



```
@startuml
skinparam backgroundColor #FFFFFF
skinparam ActivityBackgroundColor #F9F9F9
skinparam ActivityBorderColor #263238
skinparam ArrowColor #455A64
skinparam DefaultFontName "Arial"
skinparam DefaultFontSize 12

title Flux d'Activité : Phase 4 - Optimisation Bayésienne (Optuna)

start

partition "Initialisation" {
    :Charger les données (Train/Eval);
    :Récupérer le Champion de la Phase 3;
    note right: XGBoost, LGBM, etc.
}

partition "Optimisation Optuna (Étude)" {
    :Initialiser l'étude Optuna;
    note left: Direction: Maximiser F2-Score
    
    repeat
        :Trial : Suggérer Hyperparamètres;
        note right: Espace de recherche défini dans\nget_optuna_search_space()
        
        :Cross-Validation (5-Folds);
        :Calculer la métrique F2 moyenne;
        
        if (Pruning?) then (Oui)
            :Interrompre le Trial;
        else (Non)
            :Enregistrer le score du Trial;
        endif
    repeat while (n_trials ou timeout atteint?) is (Non)
}

partition "Finalisation & Métier" {
    :Extraire les Best Params;
    :Entraînement Final du Modèle;
    
    :Optimiser le Seuil (Threshold);
    note right: Recherche du point optimal pour\nminimiser le coût bancaire (FN vs FP)
    
    :Calculer les Métriques de Validation;
}

partition "Persistance (Traçabilité)" {
    :Enregistrer les résultats dans MLflow;
    note left: Tag phase="4_optimization"
    
    :Mettre à jour la table PostgreSQL;
    note right: UPDATE optimal_threshold
}

stop

footer Projet Prêt à Dépenser - Architecture MLOps
@enduml
```



*************************************************

@startuml
title Étape 0 : Logique de Configuration (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Définir l'URI de tracking;
note right: MLflowConfig.TRACKING_URI

:Interroger le serveur MLflow;
if (L'expérience existe déjà ?) then (oui)
    :Récupérer l'ID existant;
    :Log "Expérience existante";
else (non)
    :Définir le chemin des artefacts;
    :Créer l'expérience en base;
    :Récupérer le nouvel ID;
    :Log "Expérience créée";
endif

:Initialiser MlflowClient;
stop
@enduml


@startuml
title Étape 0 : Protocole de Communication (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
entity "Serveur MLflow" as MLF

[-> P4 : step0_setup()
activate P4

P4 -> MLF : mlflow.set_tracking_uri(uri)
P4 -> MLF : get_experiment_by_name(name)
activate MLF
MLF --> P4 : Experiment Object (or None)
deactivate MLF

alt Si Expérience est None
    P4 -> MLF : create_experiment(name, artifacts)
    activate MLF
    MLF --> P4 : new_id
    deactivate MLF
else Si Expérience existe
    P4 -> P4 : Extraire experiment_id
end

P4 -> P4 : Initialiser MlflowClient()
[<-- P4 : Configuration terminée
deactivate P4
@enduml


*************************************************


@startuml
title Étape 1 : Logique de Sélection du Champion (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Début de step1_load_champion();

if (¿champion_name est fourni en argument?) then (non)
    partition "_detect_champion()" {
        :Scanner le dossier /models;
        :Lire chaque fichier *_metadata.json;
        if (¿Trouvé un fichier avec is_best=True?) then (oui)
            :Extraire le model_name;
            :Retourner le nom du champion;
        else (non)
            #pink:Erreur : Aucun champion détecté;
            stop
        endif
    }
else (oui)
    :Utiliser le nom fourni;
endif

:Construire les chemins .joblib et .json;

if (¿Les fichiers existent sur le disque?) then (oui)
    :Charger le modèle (joblib.load);
    :Charger les métadonnées (json.loads);
    :Afficher le score F2 d'origine;
else (non)
    #pink:Lancer FileNotFoundError;
    stop
endif

stop
@enduml




@startuml
title Étape 1 : Chargement des Artéfacts (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
database "Système de Fichiers\n(Dossier /models)" as FS

[-> P4 : step1_load_champion()
activate P4

alt Si champion_name est None
    P4 -> FS : lister fichiers *_metadata.json
    activate FS
    FS --> P4 : Liste de fichiers
    deactivate FS
    loop Pour chaque fichier meta
        P4 -> FS : lire contenu JSON
        FS --> P4 : meta_dict
        P4 -> P4 : vérifier if meta['is_best'] == True
    end
end

== Chargement Physique ==

P4 -> FS : joblib.load(champion_model.joblib)
activate FS
FS --> P4 : Objet Modèle sklearn
deactivate FS

P4 -> FS : leer champion_metadata.json
activate FS
FS --> P4 : Dict Métadonnées
deactivate FS

P4 -> P4 : Log "Champion chargé avec succès"
[<-- P4 : Champion prêt
deactivate P4
@enduml


*************************************************

@startuml
title Étape 2 : Logique de Chargement et Préparation (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Appel de step2_load_data();

if (¿Source == "db" ET DB_AVAILABLE?) then (oui)
    partition "_load_from_db()" {
        :Connexion via get_engine();
        :Exécuter SELECT sur v_features_engineering;
        if (¿Succès?) then (oui)
            :Charger df_raw;
        else (non)
            :Log Warning "DB indisponible";
            goto load_csv;
        endif
    }
else (non)
    label load_csv
    partition "_load_from_csv()" {
        :Lire X_train.csv et y_train.csv;
        :Fusionner en df_raw;
    }
endif

partition "_split_data()" {
    :Nettoyage (remplacer Inf par NaN, fillna(0));
    :Supprimer colonnes non-numériques;
    :Appliquer train_test_split;
    note right
        **Crucial** : random_state fixe
        pour reproduire le split Phase 3
    end note
}

:Vérifier ratio de classes (stratify);
stop
@enduml




@startuml
title Étape 2 : Flux d'Orchestration des Données (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
database "PostgreSQL" as DB
database "Système de Fichiers\n(data/processed/)" as FS

[-> P4 : step2_load_data()
activate P4

alt source == "db"
    P4 -> DB : get_engine()
    P4 -> DB : pd.read_sql(query)
    activate DB
    alt Connexion OK
        DB --> P4 : DataFrame (df_raw)
    else Erreur DB
        DB --[#red]> P4 : Exception
        P4 -> P4 : Log "Bascule vers CSV"
        P4 -> FS : lire X_train.csv / y_train.csv
        activate FS
        FS --> P4 : DataFrames
        deactivate FS
    end
    deactivate DB
else source == "csv"
    P4 -> FS : lire X_train.csv / y_train.csv
    activate FS
    FS --> P4 : DataFrames
    deactivate FS
end

== Préparation Interne ==

P4 -> P4 : _split_data()
note over P4 : Nettoyage des Inf/NaN\nSélection des features
P4 -> P4 : train_test_split(stratify=y)

[<-- P4 : X_train, X_eval, y_train, y_eval prêts
deactivate P4
@enduml



*************************************************


@startuml
title Étape 3 : Logique de GridSearchCV (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Début de step3_gridsearch();

if (¿engine == "gridsearch" OU "both"?) then (oui)
    :Récupérer la grille de paramètres;
    note right: Définie selon le type de modèle\n(Logistic, RF, XGB, etc.)

    partition "Exécution Sklearn" {
        :Initialiser GridSearchCV;
        :Paramétrer Scoring = F2-Score;
        :Lancer le Fit sur X_train (CV folds);
    }

    partition "MLflow Tracking" {
        :Ouvrir un run MLflow "gridsearch";
        :Log des meilleurs hyperparamètres;
        :Log des métriques de validation (CV);
        :Enregistrer l'artefact du modèle;
    }

    :Identifier best_params y best_score;
    :Retourner le dictionnaire de résultats;
else (non)
    :Saut de l'étape;
endif

stop
@enduml



@startuml
title Étape 3 : Orchestration GridSearchCV & MLflow (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
participant "Sklearn\nGridSearchCV" as GS
entity "MLflow\nServer" as MLF

[-> P4 : step3_gridsearch()
activate P4

P4 -> GS : fit(X_train, y_train)
activate GS

loop Pour chaque combinaison (n_trials)
    GS -> GS : Cross-Validation (K-Folds)
end

GS --> P4 : results (best_estimator_)
deactivate GS

== Tracking des Résultats ==

P4 -> MLF : start_run(run_name="gridsearch")
activate MLF
P4 -> MLF : log_params(best_params)
P4 -> MLF : log_metrics(mean_test_score)
P4 -> MLF : log_model(best_estimator)
MLF --> P4 : run_id
deactivate MLF

P4 -> P4 : Stocker run_id en mlflow_run_ids
[<-- P4 : Dictionnaire de résultats (GS)
deactivate P4
@enduml


*************************************************


@startuml
title Étape 4 : Logique d'Optimisation Optuna (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Début de step4_optuna();

if (¿engine == "optuna" OU "both"?) then (oui)
    :Créer une **Study** Optuna;
    :Définir la direction : **maximize** (F2-Score);
    
    partition "Boucle d'Optimisation (n_trials)" {
        repeat
            :Suggérer hyperparamètres (TPE Sampler);
            :Initialiser Cross-Validation (K-Folds);
            
            partition "Objective Function" {
                :Entraîner modèle sur Fold k;
                :Calculer Score intermédiaire;
                if (¿Le score est médiocre?) then (oui)
                    #pink:Pruning (Arrêt précoce du trial);
                    detach
                endif
            }
            
            :Calculer F2-Score moyen;
            :Mettre à jour le modèle de probabilité (TPE);
        repeat while (¿n_trials atteint OU timeout?) is (non)
    }

    partition "MLflow Tracking" {
        :Ouvrir un run MLflow "optuna";
        :Log best_params et best_f2;
        :Log courbes d'importance des features;
    }
    
    :Retourner les meilleurs résultats;
else (non)
    :Saut de l'étape;
endif

stop
@enduml





@startuml
title Étape 4 : Orchestration Optuna, Trials et MLflow (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
participant "Optuna Study" as Study
participant "Objective Function" as Obj
entity "MLflow Server" as MLF

[-> P4 : step4_optuna()
activate P4

P4 -> Study : create_study(direction='maximize')
P4 -> Study : optimize(n_trials, timeout)
activate Study

loop Pour i de 1 à n_trials
    Study -> Obj : trial(params_suggested)
    activate Obj
    
    Obj -> Obj : cross_val_score(F2)
    
    alt Si performance insuffisante
        Obj --[#red]> Study : raise TrialPruned()
    else Trial réussi
        Obj --> Study : final_f2_score
    end
    deactivate Obj
end

Study --> P4 : best_trial (params + value)
deactivate Study

== Logging Final ==

P4 -> MLF : start_run(run_name="optuna")
activate MLF
P4 -> MLF : log_params(best_params)
P4 -> MLF : log_metrics(best_f2)
P4 -> MLF : log_artifact(optuna_optimization_history.png)
MLF --> P4 : run_id
deactivate MLF

[<-- P4 : Resultados de Optuna listos
deactivate P4
@enduml

*************************************************


@startuml
title Étape 5 : Logique de Comparaison (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Appel de step5_compare_engines();

:Récupérer les scores F2 de :
- GridSearchCV (si exécuté)
- Optuna (si exécuté);

if (¿GridSearch Score > Optuna Score?) then (oui)
    :Sélectionner **GridSearchCV** comme vainqueur;
    :best_params = gs_results['params'];
    :best_engine = "gridsearch";
else (non)
    :Sélectionner **Optuna** como vainqueur;
    :best_params = optuna_results['params'];
    :best_engine = "optuna";
endif

partition "Refit Final" {
    :Instancier le modèle avec **best_params**;
    :Entraîner le modèle sur la totalité de **X_train**;
    note right: On ne garde pas le modèle de la CV,\non ré-entraîne pour maximiser la donnée.
    :Stocker le modèle final dans **self.best_model**;
}

:Log "Le moteur [X] a gagné avec F2=[Y]";
stop
@enduml


@startuml
title Étape 5 : Orchestration du Choix Final (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
participant "Sklearn Model" as Model

[-> P4 : step5_compare_engines()
activate P4

P4 -> P4 : Extraire scores de self.gs_result et self.optuna_result

note over P4 : Logique de comparaison F2
P4 -> P4 : Déterminer best_engine

== Refit sur X_train complet ==

P4 -> Model : clone(champion_model)
activate Model
P4 -> Model : set_params(best_params)
P4 -> Model : fit(X_train, y_train)
note right : Utilise 100% du set d'entraînement\nsans découpage de validation.
Model --> P4 : best_model_fitted
deactivate Model

P4 -> P4 : Assigner best_model
[<-- P4 : Champion Phase 4 prêt
deactivate P4
@enduml




**************************************$

@startuml
title Étape 6 : Optimisation du Seuil Métier (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Calcul des probabilités sur **X_eval**;
note right: y_proba = model.predict_proba()

:Définir la fonction de coût;
note right: Coût = (FN * 5) + (FP * 1)

partition "Balayage des Seuils (Threshold Scan)" {
    :Tester 100 seuils entre 0.05 et 0.80;
    repeat
        :Appliquer le seuil courant sur y_proba;
        :Générer la Matrice de Confusion;
        :Calculer Coût, F2-Score, Précision, Rappel;
        :Stocker les résultats dans un DataFrame;
    repeat while (¿Reste des seuils?) is (oui)
}

:Identifier le seuil qui **minimise le Coût**;
:Identifier le seuil qui **maximise le F2-Score**;

:Fixer **self.optimal_threshold**;
:Générer les courbes de performance (Metrics vs Threshold);

stop
@enduml

@startuml
title Étape 6 : Évaluation et Calcul du Seuil (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
participant "Best Model (Fitted)" as Model
participant "Metrics Library\n(Sklearn)" as SK

[-> P4 : step6_optimize_threshold()
activate P4

P4 -> Model : predict_proba(X_eval)
activate Model
Model --> P4 : y_probas (0.0 to 1.0)
deactivate Model

loop Pour chaque seuil T in [0.05, 0.80]
    P4 -> P4 : y_pred = (y_probas >= T)
    P4 -> SK : confusion_matrix(y_eval, y_pred)
    SK --> P4 : TN, FP, FN, TP
    P4 -> P4 : Calculer Score F2 et Coût Métier
end

P4 -> P4 : Déterminer T_optimum (Min Coût)
note over P4 : Ce seuil devient la référence\npour la mise en production.

[<-- P4 : optimal_threshold y threshold_df prêts
deactivate P4
@enduml


**************************************$


@startuml
title Étape 7 : Logique de Sauvegarde (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Définition des chemins de fichiers;
note right: Dossiers /models et /reports

partition "Sérialisation du Modèle" {
    :Sauvegarder best_model en .joblib;
    note right: Inclut les hyperparamètres optimisés
}

partition "Métadonnées et Métriques" {
    :Calculer les métriques finales au seuil optimal;
    :Créer un dictionnaire JSON;
    note right: ID, Run_ID, Seuil, F2, Coût, etc.
    :Sauvegarder en _metadata.json;
}

partition "Courbes de Décision" {
    :Exporter le DataFrame des seuils en .csv;
    note right: Utilisé pour tracer la courbe\nCoût vs Seuil dans le Dashboard
}

:Log "Artefacts sauvegardés avec succès";
stop
@enduml



@startuml
title Étape 7 : Flux d'écriture des Artefacts (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
database "File System\n(Local Disk)" as FS

[-> P4 : step7_save()
activate P4

== Sauvegarde du Modèle ==
P4 -> FS : joblib.dump(self.best_model, "phase4_best_model.joblib")
activate FS
FS --> P4 : Confirmation OK
deactivate FS

== Sauvegarde des Métadonnées ==
P4 -> P4 : Préparer meta_dict (Seuil + Métriques)
P4 -> FS : write_text("phase4_best_model_metadata.json")
activate FS
FS --> P4 : Fichier JSON créé
deactivate FS
   
== Sauvegarde du Rapport ==
P4 -> FS : self.threshold_df.to_csv("phase4_threshold_curve.csv")
activate FS
FS --> P4 : Rapport CSV prêt
deactivate FS

[<-- P4 : Persistance terminée
deactivate P4
@enduml


**************************************$

@startuml
title Étape 8 : Logique d'Enregistrement DB (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Appel de step8_register_db();

if (¿DB_AVAILABLE est True?) then (oui)
    :Générer le nom de version;
    note right: v_phase4_YYYYMMDD_HHMMSS
    
    :Préparer le dictionnaire **info**;
    note right: Nom, Version, Run_ID, Algo, \nParams, Metrics, Paths...

    partition "Transaction PostgreSQL" {
        :Ouvrir connexion via get_engine();
        :Préparer requête INSERT INTO model_versions;
        :Caster Params et Metrics en **JSONB**;
        
        if (¿Exécution réussie?) then (oui)
            :Récupérer l'ID généré (RETURNING id);
            :Stocker **self.model_version_id**;
            :Log "Enregistré en DB avec succès";
        else (non)
            #pink:Log Erreur DB;
            :Afficher Traceback;
        endif
    }
else (non)
    :Log Warning "DB non disponible";
    :Saut de l'étape;
endif

stop
@enduml



@startuml
title Étape 8 : Protocole d'Insertion SQL (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
database "PostgreSQL\n(Table model_versions)" as DB

[-> P4 : step8_register_db()
activate P4

P4 -> DB : get_engine()
P4 -> P4 : Créer objet text(SQL_INSERT)

group Transaction SQL
    P4 -> DB : engine.begin()
    activate DB
    P4 -> DB : execute(sql, info)
    note right : Les métriques sont envoyées\nen format JSONB
    DB --> P4 : ResultProxy (id)
    P4 -> DB : commit()
    deactivate DB
end

P4 -> P4 : Assigner model_version_id
[<-- P4 : Modèle officiellement enregistré
deactivate P4
@enduml

**************************************$

@startuml
title Étape 9 : Logique de Monitoring (Action)
skinparam ActivityBackgroundColor #FEFECE

start
:Appel de step9_register_monitoring();

if (¿model_version_id est présent?) then (oui)
    :Récupérer les métriques du seuil optimal;
    note right: F2, Précision, Rappel, Coût
    
    :Préparer le tuple de monitoring;
    note right: version_id + métriques + timestamp

    partition "Enregistrement de la Baseline" {
        :Connexion à PostgreSQL;
        :INSERT INTO monitoring_metrics;
        if (¿Succès?) then (oui)
            :Log "Metrics de monitoring enregistrées";
        else (non)
            #pink:Log Warning "Échec monitoring";
        endif
    }
else (non)
    #pink:Erreur : ID de version manquant;
    stop
endif

:Pipeline Phase 4 Terminé avec Succès;
stop
@enduml


@startuml
title Étape 9 : Finalisation et Archivage (Séquence)
skinparam style strictuml

participant "Phase4Pipeline" as P4
database "PostgreSQL\n(Table monitoring)" as DB

[-> P4 : step9_register_monitoring()
activate P4

P4 -> P4 : Préparer monitoring_record
note over P4 : Liaison avec model_version_id\ncréé à l'étape 8

P4 -> DB : execute(INSERT_MONITORING)
activate DB
DB --> P4 : Success
deactivate DB

P4 -> P4 : _log("Pipeline Complete", "SUCCESS")
[<-- P4 : Fin du processus
deactivate P4
@enduml


*****************************
*****************************


@startuml
title Phase 4 : Flux d'Action Global (Vue Architecte)
skinparam ActivityBackgroundColor #FEFECE

start

partition "Initialisation" {
    :Configuration MLflow;
    :Identification du Champion Phase 3;
}

partition "Data Management" {
    :Extraction depuis PostgreSQL;
    :Split Stratifié (Train/Eval);
}

partition "Optimisation (Dual Engine)" {
    fork
        :GridSearchCV (Exhaustif);
    fork again
        :Optuna (Bayésien);
    end fork
    :Comparaison des performances;
    :Refit du Meilleur Modèle;
}

partition "Business Alignment" {
    :Optimisation du Seuil (Coût Métier);
}

partition "Persistance & Gouvernance" {
    :Sauvegarde des Artefacts;
    :Enregistrement model_versions (SQL);
    :Baseline de Monitoring (SQL);
}

stop
@enduml





