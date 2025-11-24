import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot 8.0 Mental", layout="wide")

# --- GESTION DES DONNÉES ---
if 'coups' not in st.session_state:
    st.session_state['coups'] = []

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

CLUBS_ORDER = [
    "Driver", "Bois 5", "Hybride", 
    "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", 
    "PW", "50°", "55°", "60°", 
    "Putter"
]

# Estimation rapide des distances pour pré-remplir (UX)
DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- SIDEBAR : OPTIONS ---
st.sidebar.title("⚙️ Options")

# Générateur de Test mis à jour avec les intentions
if st.sidebar.button("Générer Données Test V8"):
    new_data = []
    for _ in range(50):
        club = np.random.choice(["Driver", "Fer 7", "PW"])
        dist_target = DIST_REF[club]
        
        # Simulation : On rate souvent la distance cible
        dist_real = np.random.normal(dist_target - 5, 10) 
        
        new_data.append({
            'date': str(datetime.date.today()),
            'mode': 'Parcours',
            'club': club,
            # Intention
            'strat_type': "Départ" if club == "Driver" else "Attaque de Green",
            'strat_dist': dist_target,
            'strat_effet': "Tout droit",
            # Réalisation
            'distance': round(dist_real, 1),
            'contact': np.random.choice(["Bon Contact", "Gratte"]),
            'score_lateral': np.random.randint(0, 3),
            'direction': np.random.choice(["Centre", "Gauche"]),
            'type_coup': 'Jeu Long'
        })
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données V8 générées !")

# Export
if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v8.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot 8.0 : L'Intention")

tab_saisie, tab_analyse = st.tabs(["🧠 Saisie Stratégique", "📊 Analyse Performance"])

# --- ONGLET 1 : SAISIE EN DEUX TEMPS ---
with tab_saisie:
    
    # En-tête contextuel
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    mode_db = "Parcours" if "Parcours" in mode else "Practice"
    dist_ref_val = DIST_REF.get(club, 100)

    st.markdown("---")

    # --- SECTION 1 : L'OBJECTIF (AVANT LE COUP) ---
    st.subheader("1️⃣ L'Objectif (Avant de taper)")
    
    if club == "Putter":
        c_o1, c_o2 = st.columns(2)
        with c_o1:
            obj_dist = st.number_input("Distance à parcourir (m)", 0.0, 30.0, 3.0, 0.1)
        with c_o2:
            obj_type = st.selectbox("Type de Putt", ["Putt pour Par/Birdie", "Lag Putt (Approche)", "Putt court"])
        obj_effet = "Aucun"
    
    else:
        c_o1, c_o2, c_o3 = st.columns(3)
        with c_o1:
            # On pré-remplit avec la distance théorique du club pour gagner du temps
            obj_dist = st.number_input("Distance Cible (m)", 0, 350, dist_ref_val)
        with c_o2:
            obj_type = st.selectbox("Type de Coup", ["Départ", "Attaque de Green", "Lay-up (Sécurité)", "Approche", "Sortie de rough"])
        with c_o3:
            obj_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade (G-D)", "Draw (D-G)", "Balle basse", "Balle haute"])

    st.markdown("---")

    # --- SECTION 2 : LE RÉSULTAT (APRÈS LE COUP) ---
    st.subheader("2️⃣ Le Résultat (Après le coup)")
    
    if club == "Putter":
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1: 
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté"])
        with c_r2:
            contact = "Bon" # Moins pertinent au putt
            pente = st.selectbox("Pente réelle", ["Plat", "G-D", "D-G"])
        with c_r3:
             # On garde une logique de distance parcourue si raté
             dist_real = obj_dist if res_putt == "Dans le trou" else st.number_input("Distance parcourue réelle", 0.0, 30.0, obj_dist)
        
        if st.button("Valider Putt"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': mode_db, 'club': club,
                'strat_dist': obj_dist, 'strat_type': obj_type, # Intention
                'distance': dist_real, 'resultat': res_putt, 'pente': pente, # Réalité
                'type_coup': 'Putt', 'contact': 'Bon', 'score_lateral': 0
            })
            st.success("Putt enregistré")

    else:
        # JEU LONG
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
        with c_r2:
            contact = st.selectbox("Qualité Contact", ["Bon Contact", "Gratte", "Top", "Pointe", "Talon"])
        with c_r3:
            direction = st.radio("Direction Finale", ["Gauche", "Centre", "Droite"], horizontal=True)
            if direction != "Centre":
                score_lat = st.slider("Écart (1=Petit, 5=Perdue)", 1, 5, 2)
            else: score_lat = 0
            
        if st.button(f"Valider {club}"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': mode_db, 'club': club,
                'strat_dist': obj_dist, 'strat_type': obj_type, 'strat_effet': obj_effet, # Intention
                'distance': dist_real, 'contact': contact, 'direction': direction, # Réalité
                'score_lateral': score_lat, 'type_coup': 'Jeu Long'
            })
            st.success("Coup complet enregistré")

# --- ONGLET 2 : ANALYSE GAP ---
with tab_analyse:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long']
        
        if not df_long.empty and 'strat_dist' in df_long.columns:
            st.header("🎯 Intention vs Réalité")
            
            # Calcul du Delta (Différence)
            df_long['delta_dist'] = df_long['distance'] - df_long['strat_dist']
            
            col_kpi1, col_kpi2 = st.columns(2)
            
            avg_gap = df_long['delta_dist'].mean()
            col_kpi1.metric("Erreur de Distance Moyenne", f"{avg_gap:.1f}m", delta_color="off")
            
            if avg_gap < -5:
                col_kpi1.warning("📉 Vous êtes souvent **plus court** que prévu. Surestimez-vous vos distances ?")
            elif avg_gap > 5:
                col_kpi1.warning("🚀 Vous êtes souvent **plus long** que prévu.")
            else:
                col_kpi1.success("✅ Vous connaissez bien vos distances.")

            # Analyse par Type de Coup Stratégique
            if 'strat_type' in df_long.columns:
                st.subheader("Performance par Stratégie")
                # On considère un coup 'Réussi' si contact Bon ET dispersion < 2
                df_long['reussite'] = ((df_long['contact'] == 'Bon Contact') & (df_long['score_lateral'] < 2))
                
                stats_strat = df_long.groupby('strat_type')['reussite'].mean() * 100
                st.bar_chart(stats_strat)
                st.caption("% de coups réussis selon votre intention de départ.")
                
        else:
            st.info("Saisissez des coups avec la V8 pour voir l'analyse d'intention.")
            
    else:
        st.info("En attente de données.")
