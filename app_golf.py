import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot 6.0 Reality Check", layout="wide")

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

# --- SIDEBAR : GÉNÉRATEUR INTELLIGENT ---
st.sidebar.title("⚙️ Options")

if st.sidebar.button("Générer Données Comparatives (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(10)]
    
    for _ in range(100):
        club = np.random.choice(["Driver", "Fer 7", "PW"]) # Focus sur quelques clubs
        mode = np.random.choice(["Practice", "Parcours"])
        d = np.random.choice(dates)
        
        # Simulation : On tape plus fort et mieux au Practice
        base_dist = 150 if club == "Fer 7" else (230 if club == "Driver" else 100)
        if mode == "Practice":
            dist = np.random.normal(base_dist + 5, 5) # +5m au practice
            score_lat = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]) # Plus précis
        else:
            dist = np.random.normal(base_dist - 2, 10) # Moins régulier sur parcours
            score_lat = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])

        direction = np.random.choice(["Gauche", "Centre", "Droite"])
        if score_lat == 0: direction = "Centre"
        if direction == "Centre": score_lat = 0
        
        new_data.append({
            'date': str(d), 'mode': mode, 'club': club,
            'distance': round(dist, 1), 
            'direction': direction,
            'score_lateral': score_lat, # Note 0-5
            'longueur': "Ok", 'type_coup': 'Jeu Long',
            'resultat': f"Dispersion: {score_lat}"
        })
    
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données comparatives générées !")

# Gestion Fichiers
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Charger CSV", type="csv")
if uploaded_file:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.session_state['coups'] = df_loaded.to_dict('records')
        st.sidebar.success("Chargé !")
    except: st.sidebar.error("Erreur")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v6.csv", "text/csv")
    if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot 6.0 : Practice vs Réalité")

tab_saisie, tab_duel, tab_dispersion = st.tabs(["📝 Saisie Avancée", "⚔️ Duel: Practice vs Parcours", "🎯 Graphique Dispersion"])

# --- ONGLET 1 : SAISIE AVEC ÉCHELLE 0-5 ---
with tab_saisie:
    col_mode, col_club = st.columns(2)
    with col_mode:
        mode_active = st.radio("Où êtes-vous ?", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
        mode_db = "Parcours" if "Parcours" in mode_active else "Practice"
    with col_club:
        club = st.selectbox("Club", CLUBS_ORDER)

    st.markdown("---")
    
    if club == "Putter":
        st.info("Saisie Putt standard (voir versions précédentes pour détails)")
        # Saisie simplifiée pour l'exemple Putt
        if st.button("Enregistrer Putt"):
             st.session_state['coups'].append({'date': str(datetime.date.today()), 'mode': mode_db, 'club': 'Putter', 'type_coup': 'Putt', 'distance': 1, 'score_lateral':0})
             st.success("Putt noté")
    else:
        # Saisie JEU LONG
        c1, c2 = st.columns(2)
        with c1:
            dist = st.number_input("Distance (m)", 0, 320, 140)
        with c2:
            direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
        
        # LA NOUVEAUTÉ : SLIDER 0-5
        if direction != "Centre":
            score_lat = st.slider("🔴 Sévérité de l'écart (0 = Axe, 5 = Catastrophe)", 1, 5, 2)
            st.caption("1: Bord de Green | 3: Rough/Bunker | 5: Hors Limites/Perdu")
        else:
            score_lat = 0
            st.success("✅ Plein Axe (Score: 0)")

        if st.button(f"Valider {club} ({mode_db})", type="primary"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()),
                'mode': mode_db,
                'club': club,
                'distance': dist,
                'direction': direction,
                'score_lateral': score_lat, # Le fameux 0-5
                'type_coup': 'Jeu Long',
                'longueur': 'Ok' # Simplifié pour cet exemple
            })
            st.success(f"Coup enregistré avec score de dispersion : {score_lat}")

# --- ONGLET 2 : COMPARATEUR PRACTICE vs PARCOURS ---
with tab_duel:
    if not st.session_state['coups']:
        st.info("Entrez des données dans les deux modes pour comparer.")
    else:
        df = pd.DataFrame(st.session_state['coups'])
        df = df[df['type_coup'] == 'Jeu Long'] # On analyse pas le putt ici
        
        st.header("⚔️ La Vérité du Terrain")
        
        club_comp = st.selectbox("Choisir le club à comparer", df['club'].unique())
        
        subset = df[df['club'] == club_comp]
        
        # Séparation des données
        df_prac = subset[subset['mode'] == 'Practice']
        df_parc = subset[subset['mode'] == 'Parcours']
        
        if not df_prac.empty and not df_parc.empty:
            col_a, col_b, col_c = st.columns(3)
            
            # 1. Comparaison Distance
            avg_prac = df_prac['distance'].mean()
            avg_parc = df_parc['distance'].mean()
            delta_dist = avg_parc - avg_prac
            
            col_a.metric("Distance Practice", f"{avg_prac:.1f}m")
            col_a.metric("Distance Parcours", f"{avg_parc:.1f}m", f"{delta_dist:.1f}m", delta_color="normal")
            
            # 2. Comparaison Dispersion Moyenne (Note 0-5)
            disp_prac = df_prac['score_lateral'].mean()
            disp_parc = df_parc['score_lateral'].mean()
            delta_disp = disp_prac - disp_parc # Si négatif, c'est que parcours est pire (chiffre plus haut)
            
            col_b.metric("Dispersion Practice (0-5)", f"{disp_prac:.1f}")
            col_b.metric("Dispersion Parcours (0-5)", f"{disp_parc:.1f}", f"{delta_disp:.1f}", delta_color="inverse")
            
            # 3. Analyse
            with col_c:
                st.write("### Analyse")
                if avg_prac > avg_parc + 10:
                    st.warning(f"⚠️ Vous perdez **{int(avg_prac - avg_parc)}m** sur le parcours ! Probablement un manque d'engagement ou des lies difficiles.")
                elif disp_parc > disp_prac + 1:
                    st.error("⚠️ Votre dispersion explose sur le parcours. Le stress ou l'alignement sont en cause.")
                else:
                    st.success("👏 Vos performances sont transférées efficacement du practice au parcours.")
        else:
            st.warning(f"Il faut des données de '{club_comp}' à la fois en Practice et sur Parcours.")

# --- ONGLET 3 : GRAPHIQUE DISPERSION AVANCÉ ---
with tab_dispersion:
    st.header("🎯 Analyse Fine de la Dispersion")
    
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_graph = df[df['type_coup'] == 'Jeu Long']
        
        filter_club = st.selectbox("Club pour Graphique", df_graph['club'].unique(), key="graph_club")
        data_plot = df_graph[df_graph['club'] == filter_club]
        
        if not data_plot.empty:
            # Préparation du Graphique
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calcul des coordonnées X (Latéral) basé sur le score 0-5
            # Si Gauche : X = -Score. Si Droite : X = +Score. Si Centre : X = 0.
            def get_x_coord(row):
                val = row['score_lateral']
                # Petit bruit aléatoire pour éviter que les points se montent dessus
                jitter = np.random.normal(0, 0.15) 
                if row['direction'] == 'Gauche': return (val * -1) + jitter
                if row['direction'] == 'Droite': return val + jitter
                return 0 + jitter

            data_plot['x_val'] = data_plot.apply(get_x_coord, axis=1)
            
            # On sépare les couleurs par Mode
            prac_p = data_plot[data_plot['mode'] == 'Practice']
            parc_p = data_plot[data_plot['mode'] == 'Parcours']
            
            # Plot Practice (Bleu, ronds)
            ax.scatter(prac_p['x_val'], prac_p['distance'], c='blue', alpha=0.5, label='Practice', s=80, edgecolors='white')
            # Plot Parcours (Rouge, croix)
            ax.scatter(parc_p['x_val'], parc_p['distance'], c='red', alpha=0.7, marker='X', label='Parcours', s=100)
            
            # Esthétique
            ax.axvline(0, color='green', linestyle='--', alpha=0.5, label="Axe Idéal")
            ax.set_xlabel("⟵ Gauche (Score 5-1)  |  Centre (0)  |  Droite (Score 1-5) ⟶")
            ax.set_ylabel("Distance (mètres)")
            ax.set_xlim(-6, 6) # Echelle fixe pour bien voir le 0-5
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.caption("Les croix ROUGES sont vos coups sur le parcours. Les ronds BLEUS au practice. Plus on s'éloigne du centre (0), plus le coup est raté.")
            
        else:
            st.write("Pas de données pour ce club.")
