# =============================================================================
# monitoring/simulator.py — Versión Actualizada (20 Features)
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Configuration de l'URL API ---
URL_API = "http://localhost:8001/predict"

def main():
    st.set_page_config(page_title="Simulateur de Crédit Pro", layout="wide")

    st.title("🏦 Simulateur de Score Crédit (Modèle 20 Features)")
    st.markdown("Saisissez los datos del cliente para obtener la probabilidad de default y explicaciones SHAP.")

    # --- FORMULAIRE D'ENTRÉE (Ajustado a ClientDataInput) ---
    with st.sidebar:
        st.header("📋 Información del Cliente")
        with st.form("credit_form"):

            st.subheader("⭐ Variables Críticas (Scores)")
            ext_1 = st.slider("EXT_SOURCE_1", 0.01, 0.96, 0.5)
            ext_2 = st.slider("EXT_SOURCE_2", 0.0, 0.85, 0.5)
            ext_3 = st.slider("EXT_SOURCE_3", 0.0, 0.90, 0.5)

            st.divider()

            st.subheader("💰 Datos Financieros")
            amt_credit = st.number_input("Montant Crédit (€)", 45000.0, 2000000.0, 500000.0)
            amt_annuity = st.number_input("Annuité (€)", 1600.0, 100000.0, 25000.0)
            goods_price = st.number_input("Prix du Bien (€)", 40000.0, 2000000.0, 450000.0)

            st.divider()

            st.subheader("📊 Historial y Comportamiento")
            pay_ratio = st.slider("Ratio de Paiement", 0.0, 1.0, 0.1)
            pay_delay = st.number_input("Délai Moyen (jours)", 0.0, 100.0, 2.0)
            max_dpd = st.number_input("Max Days Past Due", 0.0, 365.0, 0.0)

            st.divider()

            st.subheader("👤 Perfil Demográfico")
            age = st.number_input("Âge", 18, 70, 35)
            years_emp = st.number_input("Années d'emploi", 0, 50, 10)
            gender = st.selectbox("Genre", ["F", "M", "XNA"])
            education = st.selectbox("Éducation", [
                "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])

            st.divider()

            st.subheader("📁 Otros Datos de Buró")
            b_credit = st.number_input("Total Crédits Bureau", value=5.0)
            b_debt = st.number_input("Dette Moyenne Bureau", value=1000.0)
            pos_m = st.number_input("Mois POS moyen", value=12.0)
            cc_draw = st.number_input("Retraits CC moyen", value=0.0)
            cc_bal = st.number_input("Solde CC moyen", value=0.0)
            phone_ch = st.number_input("Jours changement phone", value=365.0)
            region = st.selectbox("Note Région", [1, 2, 3], index=1)

            submit = st.form_submit_button("🚀 Analyser le Dossier")

    # --- APPEL API ---
    if submit:
        # Construimos el payload exacto para ClientDataInput
        payload = {
            "ext_source_1": ext_1, "ext_source_2": ext_2, "ext_source_3": ext_3,
            "paymnt_ratio_mean": pay_ratio, "paymnt_delay_mean": pay_delay,
            "max_dpd": max_dpd, "age": age, "years_employed": years_emp,
            "code_gender": gender, "education_type": education,
            "amt_credit": amt_credit, "amt_annuity": amt_annuity,
            "goods_price": goods_price, "bureau_credit_total": b_credit,
            "bureau_debt_mean": b_debt, "pos_months_mean": pos_m,
            "cc_drawings_mean": cc_draw, "cc_balance_mean": cc_bal,
            "phone_change_days": phone_ch, "region_rating": region
        }

        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(URL_API, json=payload)
                response.raise_for_status()
                resultat = response.json()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Décision")
                prob = resultat["probabilite_defaut"]
                score_percent = (1 - prob) * 100

                st.metric("Confianza (Salud)", f"{score_percent:.1f}%")

                if resultat["decision"].upper() in ["ACCORD", "ACCORDÉ", "APPROVE"]:
                    st.success(f"✅ {resultat['decision']}")
                else:
                    st.error(f"❌ {resultat['decision']}")

                st.caption(f"Probabilité de défaut : {prob:.4f}")

            with col2:
                st.subheader("💡 Explication SHAP (Top Influences)")
                explicaciones = resultat.get("explication_shap", [])

                if explicaciones:
                    df_shap = pd.DataFrame(explicaciones)

                    # Gráfico de barras horizontales
                    fig = px.bar(
                        df_shap,
                        x="impact_shap",
                        y="feature",
                        orientation='h',
                        color="direction",
                        color_discrete_map={
                            "hausse_risque": "#e74c3c", # Rojo
                            "baisse_risque": "#2ecc71"  # Verde
                        },
                        hover_data=["valeur_client"],
                        title="Impacto en el Score Final"
                    )
                    # Invertimos el eje Y para que la más importante esté arriba
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Aucune explication reçue.")

        except Exception as e:
            st.error(f"Erreur : {e}")

if __name__ == "__main__":
    main()