import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot 9.0 Ultimate", layout="wide")

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

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- SIDEBAR : GÉNÉRATEUR DE TEST COMPLET ---
st.sidebar.title("⚙️ Data Lab")

if st.sidebar.button("Générer Dataset 'Pro' (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(20)]
    
    for _ in range(200):
        club = np.random.choice(["Driver", "Fer 7", "PW"])
        mode = np.random.choice(["Practice", "Parcours"])
        dist_target = DIST_REF[club]
        
        # Simulation d'erreurs réalistes
        is_practice = (mode == "Practice")
        std_dev = 5 if is_practice else 12 # Plus précis au practice
        dist_real = np.random.normal(dist_target - 3, std_dev) # Tendance à être court de 3m
        
        # Effets
        intention_effet = np.random.choice(["Tout droit", "Fade", "Draw"], p=[0.7, 0.15, 0.15])
        # Réussite de l'effet (80% au practice, 50% sur parcours)
        success_prob = 0.8 if is_practice else 0.5
        if np.random.random() < success_prob:
            real_effet = intention_effet
        else:
            real_effet = "Tout droit" if intention_effet != "Tout droit" else np.random.choice(["Fade", "Draw"])
            
        # Latéral
        lat_score = abs(np.random.normal(0, 1 if is_practice else 2))
        lat_score = min(5, int(lat_score))
        direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])

        new_data.append({
            'date': str(np.random.choice(dates)),
            'mode': mode,
            'club': club,
            'strat_dist': dist_target,
            'strat_type': "Attaque de Green",
            'strat_effet': intention_effet, # Ce que je voulais faire
            'distance': round(dist_real, 1),
            'real_effet': real_effet,       # Ce que j'ai fait
            'contact': np.random.choice(["Bon", "Gratte", "Top"], p=[0.7, 0.2, 0.1]),
            'direction': direction,
            'score_lateral': lat_score,
            'type_coup': 'Jeu Long'
        })
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("200 Coups générés pour l'analyse !")

# Export
if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Exporter CSV", convert_df(df_ex), "golf_v9.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot 9.0 : Ultimate Analytics")

tab_saisie, tab_analyse, tab_data = st.tabs(["🧠 Saisie Stratégique", "📈 Analyse 360°", "🗃️ Données"])

# --- ONGLET 1 : SAISIE STRATÉGIQUE ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    mode_db = "Parcours" if "Parcours" in mode else "Practice"
    dist_ref_val = DIST_REF.get(club, 100)

    st.markdown("---")
    col_int, col_real = st.columns(2)

    # --- COLONNE GAUCHE : INTENTION ---
    with col_int:
        st.subheader("1️⃣ L'INTENTION")
        st.info("Planifiez votre coup")
        if club == "Putter":
            obj_dist = st.number_input("Distance Cible (m)", 0.0, 30.0, 3.0, 0.1)
            obj_effet = "Aucun"
        else:
            obj_dist = st.number_input("Distance Cible (m)", 0, 350, dist_ref_val)
            obj_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade (G-D)", "Draw (D-G)", "Balle basse"])
            obj_type = st.selectbox("Stratégie", ["Attaque de Green", "Départ", "Lay-up", "Sécurité"])

    # --- COLONNE DROITE : RÉALITÉ ---
    with col_real:
        st.subheader("2️⃣ LA RÉALITÉ")
        st.warning("Notez le résultat")
        if club == "Putter":
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté"])
            dist_real = obj_dist if res_putt == "Dans le trou" else st.number_input("Dist. Parcourue", 0.0, 30.0, obj_dist)
            real_effet = "Aucun"
            contact = "Bon"
            score_lat = 0
            direction = "Centre"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            real_effet = st.selectbox("Effet Réalisé", ["Tout droit", "Fade (G-D)", "Draw (D-G)", "Hook/Slice", "Involontaire"])
            contact = st.selectbox("Contact", ["Bon", "Gratte", "Top", "Pointe/Talon"])
            
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"])
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0

    st.markdown("---")
    if st.button("💾 Enregistrer le Coup", type="primary", use_container_width=True):
        coup_data = {
            'date': str(datetime.date.today()), 'mode': mode_db, 'club': club,
            'strat_dist': obj_dist, 'strat_effet': obj_effet, # Intention
            'distance': dist_real, 'real_effet': real_effet,  # Réalité
            'contact': contact, 'direction': direction, 'score_lateral': score_lat,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        }
        st.session_state['coups'].append(coup_data)
        st.success("Données sauvegardées !")

# --- ONGLET 2 : ANALYSE 360 ---
with tab_analyse:
    if not st.session_state['coups']:
        st.info("En attente de données...")
    else:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long'].copy()
        
        if not df_long.empty:
            # --- PARTIE 1 : DISPERSION & ELLIPSES ---
            st.header("🎯 Dispersion & Ellipses de Confiance")
            
            filter_club = st.selectbox("Analyser Club", df_long['club'].unique())
            subset = df_long[df_long['club'] == filter_club]
            
            if len(subset) > 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Construction des coordonnées pour le graph
                def get_coords(row):
                    # X = Latéral (approximatif basé sur le score 0-5)
                    x_base = row['score_lateral'] * 5 # 1 point = 5m d'écart approx
                    if row['direction'] == 'Gauche': x = -x_base
                    elif row['direction'] == 'Droite': x = x_base
                    else: x = 0
                    # Ajout de bruit (jitter) pour visibilité
                    return x + np.random.normal(0, 1), row['distance']

                coords = subset.apply(get_coords, axis=1, result_type='expand')
                subset['x_viz'] = coords[0]
                subset['y_viz'] = coords[1]
                
                # Séparation Practice / Parcours
                prac = subset[subset['mode'] == 'Practice']
                parc = subset[subset['mode'] == 'Parcours']
                
                # Fonction Ellipse
                def draw_ellipse(x, y, ax, color, label):
                    if len(x) < 3: return
                    cov = np.cov(x, y)
                    lambda_, v = np.linalg.eig(cov)
                    lambda_ = np.sqrt(lambda_)
                    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                                  width=lambda_[0]*4, height=lambda_[1]*4, # 2 std dev = 95%
                                  angle=np.rad2deg(np.arccos(v[0, 0])), 
                                  edgecolor=color, facecolor=color, alpha=0.2, label=f'Zone {label}')
                    ax.add_artist(ell)
                    ax.scatter(x, y, c=color, s=30, alpha=0.6)

                draw_ellipse(prac['x_viz'], prac['y_viz'], ax, 'blue', 'Practice')
                draw_ellipse(parc['x_viz'], parc['y_viz'], ax, 'red', 'Parcours')
                
                # Cible
                target_dist = subset['strat_dist'].mean()
                ax.scatter([0], [target_dist], c='green', s=200, marker='*', label='Cible Moyenne')
                
                ax.set_xlabel("← Gauche (m) | Droite (m) →")
                ax.set_ylabel("Distance (m)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.autoscale()
                st.pyplot(fig)
                st.caption("Les ellipses représentent la zone où 95% de vos balles atterrissent.")
            else:
                st.warning("Pas assez de données pour ce club (min 3 coups).")

            st.markdown("---")

            # --- PARTIE 2 : GAP ANALYSIS (Histogramme) ---
            st.header("📏 Gap Analysis : Intention vs Réalité")
            
            # Calcul du delta
            df_long['delta'] = df_long['distance'] - df_long['strat_dist']
            
            col_gap_L, col_gap_R = st.columns([2, 1])
            
            with col_gap_L:
                fig2, ax2 = plt.subplots()
                sns.histplot(df_long['delta'], kde=True, ax=ax2, color="purple", bins=15)
                ax2.axvline(0, color='green', linestyle='--', linewidth=2, label="Distance Cible")
                ax2.axvline(df_long['delta'].mean(), color='red', linestyle='-', label="Votre Moyenne")
                ax2.set_title("Erreur de Distance (Mètres)")
                ax2.legend()
                st.pyplot(fig2)
                
            with col_gap_R:
                mean_error = df_long['delta'].mean()
                abs_error = df_long['delta'].abs().mean()
                st.metric("Erreur Moyenne (Biais)", f"{mean_error:.1f}m", 
                          help="Si négatif : Vous êtes trop court en moyenne.")
                st.metric("Dispersion (Précision)", f"{abs_error:.1f}m",
                          help="Écart moyen par rapport à la cible, qu'il soit court ou long.")
                
                if mean_error < -5:
                    st.error("🚨 DIAGNOSTIC : Vous sous-estimez vos distances. Prenez un club de plus !")

            st.markdown("---")

            # --- PARTIE 3 : MAÎTRISE DES EFFETS ---
            st.header("🎨 Maîtrise des Effets")
            
            # On ne garde que les coups où on a VOLONTAIREMENT tenté un effet
            df_effets = df_long[df_long['strat_effet'].isin(["Fade", "Draw"])]
            
            if not df_effets.empty:
                # Vérification de succès : Si j'ai demandé Draw et j'ai eu Draw
                df_effets['succes'] = df_effets.apply(lambda x: x['strat_effet'] in x['real_effet'], axis=1)
                
                success_rate = df_effets.groupby('strat_effet')['succes'].mean() * 100
                
                col_eff1, col_eff2 = st.columns(2)
                with col_eff1:
                    st.write("Taux de réussite par effet annoncé :")
                    st.bar_chart(success_rate)
                with col_eff2:
                    st.write("Détail :")
                    st.dataframe(df_effets[['club', 'strat_effet', 'real_effet', 'succes']])
            else:
                st.info("Aucun coup à effet (Fade/Draw) tenté pour l'instant.")

# --- ONGLET 3 : DONNÉES BRUTES ---
with tab_data:
    st.header("🗃️ Base de données")
    st.dataframe(df, use_container_width=True)
