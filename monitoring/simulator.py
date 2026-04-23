# =============================================================================
# monitoring/simulator.py — Version Francisée (20 Variables)
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Configuration de l'URL de l'API ---
URL_API = "http://localhost:8001/predict"

def main():
    st.set_page_config(page_title="Simulateur de Crédit Pro", layout="wide")

    st.title("🏦 Simulateur de Score Crédit (Modèle 20 Variables)")
    st.markdown("Saisissez les données du client pour obtenir la probabilité de défaut et les explications SHAP.")

    # --- FORMULAIRE D'ENTRÉE (Ajusté à ClientDataInput) ---
    with st.sidebar:
        st.header("📋 Informations Client")
        with st.form("credit_form"):

            st.subheader("⭐ Variables Critiques (Scores)")
            ext_1 = st.slider("SCORE EXTERNE 1", 0.01, 0.96, 0.5)
            ext_2 = st.slider("SCORE EXTERNE 2", 0.0, 0.85, 0.5)
            ext_3 = st.slider("SCORE EXTERNE 3", 0.0, 0.90, 0.5)

            # --- À l'intérieur du st.form ---
            st.divider()
            st.subheader("📦 Données de Situation")

            # Les 3 Sélecteurs catégoriels
            type_pret = st.selectbox("Type de prêt", options=["Cash loans", "Revolving loans"])
            objet_pret = st.selectbox("Objet du prêt", options=["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_A", "Other_B", "Group of people"])
            type_residence = st.selectbox("Type de résidence", options=["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"])

            st.divider()

            st.subheader("💰 Données Financières")
            amt_income_total = st.number_input("Revenu annuel (€)", 10000.0, 1000000.0, 50000.0)
            amt_credit = st.number_input("Montant du Crédit (€)", 45000.0, 2000000.0, 500000.0)
            amt_annuity = st.number_input("Annuité (€)", 1600.0, 100000.0, 25000.0)
            goods_price = st.number_input("Prix du Bien (€)", 40000.0, 2000000.0, 450000.0)

            st.divider()

            st.subheader("📊 Historique et Comportement")
            pay_ratio = st.slider("Ratio de Paiement", 0.0, 1.0, 0.1)
            pay_delay = st.number_input("Délai Moyen (jours)", 0.0, 100.0, 2.0)
            max_dpd = st.number_input("Jours de Retard Max (DPD)", 0.0, 365.0, 0.0)

            st.divider()

            st.subheader("👤 Profil Démographique")
            age = st.number_input("Âge", 18, 70, 35)
            years_emp = st.number_input("Années d'ancienneté", 0, 50, 10)
            gender = st.selectbox("Genre", ["F", "M", "XNA"])
            education = st.selectbox("Niveau d'Études", [
                "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])

            st.divider()

            st.subheader("📁 Autres Données (Bureau)")
            b_credit = st.number_input("Total Crédits Bureau", value=5.0)
            b_debt = st.number_input("Dette Moyenne Bureau", value=1000.0)
            pos_m = st.number_input("Moyenne mois POS", value=12.0)
            cc_draw = st.number_input("Retraits CB moyens", value=0.0)
            cc_bal = st.number_input("Solde CB moyen", value=0.0)
            phone_ch = st.number_input("Jours depuis changement tél.", value=365.0)
            region = st.selectbox("Note Région", [1, 2, 3], index=1)

            submit = st.form_submit_button("🚀 Analyser le Dossier")

    # --- APPEL À L'API ---
    if submit:
        # Construction du payload exact pour ClientDataInput
        # Le nom de la clé correspond au MAPPING_COLONNES de l'API
        payload = {
            # 1. Scores Critiques (3 variables)
            "ext_source_1": ext_1,
            "ext_source_2": ext_2,
            "ext_source_3": ext_3,

            # 2. Catégories / Sélecteurs (3 variables)
            "type_pret": type_pret,
            "objet_pret": objet_pret,
            "type_residence": type_residence,

            # 3. Données Financières (4 variables)
            "revenu": amt_income_total,
            "amt_credit": amt_credit,
            "amt_annuity": amt_annuity,
            "goods_price": goods_price,

            # 4. Historique et Comportement (3 variables)
            "paymnt_ratio_mean": pay_ratio,
            "paymnt_delay_mean": pay_delay,
            "max_dpd": max_dpd,

            # 5. Profil Démographique (4 variables)
            "age": age,
            "years_employed": years_emp,
            "code_gender": gender,
            "education_type": education,

            # 6. Autres Données / Bureau (7 variables)
            "bureau_credit_total": b_credit,
            "bureau_debt_mean": b_debt,
            "pos_months_mean": pos_m,
            "cc_drawings_mean": cc_draw,
            "cc_balance_mean": cc_bal,
            "phone_change_days": phone_ch,
            "region_rating": region
        }

        # --- DEBUG : Inspection du Payload avant l'envoi ---
        st.info("🔍 Debug : Structure du Payload envoyée à l'API")

        # Utilisation de colonnes pour optimiser l'espace vertical
        debug_col1, debug_col2 = st.columns(2)
        payload_items = list(payload.items())
        mid = len(payload_items) // 2

        with debug_col1:
            for key, value in payload_items[:mid]:
                st.write(f"**{key}** : `{value}` ({type(value).__name__})")

        with debug_col2:
            for key, value in payload_items[mid:]:
                st.write(f"**{key}** : `{value}` ({type(value).__name__})")
        # ----------------------------------------------------

        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(URL_API, json=payload)
                response.raise_for_status()
                resultat = response.json()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Décision")
                prob = resultat["probabilite_defaut"]
                score_sante = (1 - prob) * 100

                st.metric("Indice de Confiance", f"{score_sante:.1f}%")

                if resultat["decision"].upper() in ["ACCORD", "ACCORDÉ", "APPROVE", "APPROUVÉ"]:
                    st.success(f"✅ {resultat['decision']}")
                else:
                    st.error(f"❌ {resultat['decision']}")

                st.caption(f"Probabilité de défaut : {prob:.4f}")

            with col2:
                st.subheader("💡 Explication SHAP (Influences Majeures)")
                explications = resultat.get("explication_shap", [])

                if explications:
                    df_shap = pd.DataFrame(explications)
                    # Mise en minuscule pour la gestion de la direction de l'impact
                    df_shap["direction"] = df_shap["direction"].str.lower().str.replace(" ", "_")

                    # Graphique de barres horizontales
                    fig = px.bar(
                        df_shap,
                        x="impact_shap",
                        y="feature",
                        orientation='h',
                        color="direction",
                        color_discrete_map={
                            "hausse_risque": "red",     # Rouge pour l'augmentation du risque
                            "baisse_risque": "green",   # Vert pour la réduction du risque
                            "neutre": "grey"            # Gris pour les impacts nuls
                        },
                        hover_data=["valeur_client"],
                        title="Impact sur le Score Final"
                    )

                    # Inversion de l'axe Y pour placer les features les plus importantes en haut
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Aucune explication reçue.")

        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")

if __name__ == "__main__":
    main()