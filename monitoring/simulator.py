# =============================================================================
# monitoring/simulator.py — Interface de Simulation de Crédit
# Envoie des données à l'API FastAPI et affiche le score + SHAP
# =============================================================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Configuration de l'URL API ---
URL_API = "http://localhost:8001/predict"

def main():
    st.set_page_config(page_title="Simulateur de Crédit", layout="wide")

    st.title("🏦 Simulateur de Score Crédit")
    st.markdown("Saisissez les informations du client para obtenir une décision en temps réel.")

    # --- FORMULAIRE D'ENTRÉE (Basé sur ClientDataInput) ---
    with st.sidebar:
        st.header("📋 Informations Client")
        with st.form("credit_form"):
            # Features Numériques
            age = st.number_input("Âge", 18, 100, 35)
            revenu = st.number_input("Revenu Annuel (€)", min_value=0.0, value=45000.0, step=1000.0)
            montant_pret = st.number_input("Montant du Prêt (€)", min_value=0.0, value=15000.0, step=500.0)
            duree_pret_mois = st.slider("Durée (mois)", 1, 360, 48)

            st.divider()

            # Historique
            jours_retard = st.number_input("Jours de retard moyen", 0.0, 100.0, 0.5)
            taux_incid = st.slider("Taux d'incidents", 0.0, 1.0, 0.02)
            taux_util = st.slider("Utilisation Crédit", 0.0, 1.0, 0.45)
            nb_comptes = st.number_input("Nb Comptes Ouverts", 0, 50, 3)

            st.divider()

            # Catégorielles
            residence = st.selectbox("Résidence", [
                "House / apartment",
                "Rented apartment",
                "With parents",
                "Municipal apartment"
            ])
            objet = st.selectbox("Objet du prêt", ["Éducation", "Immobilier", "Personnel", "Automobile", "Médical"])
            type_p = st.radio("Type de prêt", ["Garanti", "Non garanti"])

            submit = st.form_submit_button("🚀 Analyser le Dossier")

    # --- APPEL API ET AFFICHAGE DES RÉSULTATS ---
    # --- Fragmento corregido de monitoring/simulator.py ---

    if submit:
        payload = {
            "age": age, "revenu": revenu, "montant_pret": montant_pret,
            "duree_pret_mois": duree_pret_mois, "jours_retard_moyen": jours_retard,
            "taux_incidents": taux_incid, "taux_utilisation_credit": taux_util,
            "nb_comptes_ouverts": nb_comptes, "type_residence": residence,
            "objet_pret": objet, "type_pret": type_p
        }

        try:
            with st.spinner("Analyse en cours..."):
                response = requests.post(URL_API, json=payload)
                response.raise_for_status()
                resultat = response.json()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Décision")
                # CLAVE CORREGIDA: 'probabilite_defaut' en lugar de 'probabilite'
                prob = resultat["probabilite_defaut"]
                score_percent = (1 - prob) * 100

                st.metric("Confiance (Santé Financière)", f"{score_percent:.1f}%")

                # Adaptación a "Refuse" / "Accord" (ajusta según los strings de tu API)
                if resultat["decision"].upper() in ["ACCORD", "ACCORDÉ", "APPROVE"]:
                    st.success(f"✅ {resultat['decision']}")
                else:
                    st.error(f"❌ {resultat['decision']}")

                st.caption(f"Probabilité de défaut : {prob:.4f}")
                st.caption(f"Seuil utilisé : {resultat['seuil_utilise']}")

            with col2:
                st.subheader("💡 Explication SHAP")
                # TU API ENVÍA UNA LISTA DE DICCIONARIOS EN 'explication_shap'
                explicaciones = resultat.get("explication_shap", [])

                if explicaciones:
                    # Convertimos la lista de dicts directamente a DataFrame
                    df_shap = pd.DataFrame(explicaciones)

                    # Renombramos para el gráfico
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
                        # Mostramos el valor real del cliente al pasar el ratón
                        hover_data=["valeur_client", "explication"],
                        title="Influence sur le score"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Aucune explication reçue.")

        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

if __name__ == "__main__":
    main()