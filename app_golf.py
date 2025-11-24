import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 22.0 Combine & Score", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    h4 {color: #1565C0;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# --- DONNÉES & CONSTANTES ---
if 'coups' not in st.session_state:
    st.session_state['coups'] = []
if 'parties' not in st.session_state:
    st.session_state['parties'] = []
if 'combine_state' not in st.session_state:
    st.session_state['combine_state'] = None # Pour stocker le test en cours

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

CLUBS_ORDER = [
    "Driver", "Bois 5", "Hybride", 
    "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", 
    "PW", "50°", "55°", "60°", 
    "Putter"
]

SHOT_TYPES = [
    "Départ (Tee Shot)", "Attaque de Green", "Lay-up / Sécurité", 
    "Approche (<50m)", "Sortie de Bunker", "Recovery"
]

PUTT_RESULTS = [
    "Dans le trou", "Court", "Long", "Gauche", "Droite",
    "Court-Gauche", "Court-Droite", "Long-Gauche", "Long-Droite"
]

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- BARRE LATÉRALE : DATA ---
st.sidebar.title("⚙️ Data Lab")

# IMPORT
uploaded_file = st.sidebar.file_uploader("📂 Importer Historique (CSV)", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        current_data = pd.DataFrame(st.session_state['coups'])
        combined = pd.concat([current_data, df_loaded], ignore_index=True).drop_duplicates()
        st.session_state['coups'] = combined.to_dict('records')
        st.sidebar.success(f"{len(df_loaded)} coups importés !")
    except Exception as e:
        st.sidebar.error(f"Erreur import : {e}")

st.sidebar.markdown("---")

# GÉNÉRATEUR V22
if st.sidebar.button("Générer Données V22 (Test)"):
    # Génération rapide pour peupler les tableaux
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(30)]
    for _ in range(300):
        mode = np.random.choice(["Parcours", "Practice", "Combine"], p=[0.5, 0.3, 0.2])
        club = np.random.choice(["Driver", "Fer 7", "Putter"])
        
        if club == "Putter":
            res = np.random.choice(PUTT_RESULTS, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            dist = np.random.randint(1, 10)
            new_data.append({
                'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
                'strat_dist': dist, 'distance': dist, 'resultat_putt': res, 'type_coup': 'Putt',
                'score_lateral': 0, 'direction': 'Centre'
            })
        else:
            target = DIST_REF[club]
            real = np.random.normal(target, 10)
            lat = np.random.randint(0, 4)
            direc = "Centre" if lat == 0 else np.random.choice(["Gauche", "Droite"])
            new_data.append({
                'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
                'strat_dist': target, 'distance': real, 'score_lateral': lat, 'direction': direc,
                'type_coup': 'Jeu Long', 'resultat_putt': 'N/A', 'delta_dist': real-target
            })
            
    st.session_state['coups'].extend(new_data)
    # Génération fausses parties
    st.session_state['parties'] = [
        {'date': str(dates[0]), 'score': 85, 'putts': 32, 'gir': 8, 'fairways': 6},
        {'date': str(dates[5]), 'score': 82, 'putts': 30, 'gir': 9, 'fairways': 7}
    ]
    st.sidebar.success("Données générées !")

# EXPORT
if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v22.csv", "text/csv")

if st.sidebar.button("🗑️ Reset Tout"): 
    st.session_state['coups'] = []
    st.session_state['parties'] = []
    st.session_state['combine_state'] = None

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 22.0 : Combine & Score")

tab_parcours, tab_practice, tab_combine, tab_dna, tab_sac, tab_putt = st.tabs([
    "⛳ Parcours & Score", 
    "🚜 Practice Libre",
    "🏆 Combine Test",
    "🧬 Analyse Club", 
    "🎒 Bag Mapping",
    "🟢 Putting 360"
])

# ==================================================
# ONGLET 1 : PARCOURS & CARTE DE SCORE
# ==================================================
with tab_parcours:
    col_saisie, col_carte = st.columns([1, 1])
    
    # --- A. SAISIE COUP PARCOURS ---
    with col_saisie:
        st.header("📝 Saisie Coup (Parcours)")
        club = st.selectbox("Club", CLUBS_ORDER, key="p_club")
        par_trou = st.selectbox("Par du trou", [3, 4, 5], key="p_par")
        
        c1, c2 = st.columns(2)
        with c1:
            shot_type = st.selectbox("Type", SHOT_TYPES, key="p_type")
            # Logique Tee Shot Par 4/5
            if shot_type == "Départ (Tee Shot)" and par_trou > 3:
                obj_dist = 0.0
                st.caption("🎯 Objectif : Max Distance")
            else:
                obj_dist = st.number_input("Cible (m)", 0, 350, DIST_REF.get(club, 100), key="p_obj")
                
        with c2:
            dist_real = st.number_input("Réalisé (m)", 0, 350, int(obj_dist) if obj_dist>0 else 200, key="p_real")
            real_effet = st.selectbox("Effet", ["Tout droit", "Fade", "Draw", "Raté"], key="p_eff")

        # Dispersion
        if club != "Putter":
            cc1, cc2 = st.columns(2)
            with cc1: direction = st.radio("Axe", ["Gauche", "Centre", "Droite"], horizontal=True, key="p_dir")
            with cc2: score_lat = st.slider("Écart (0-5)", 0, 5, 0, key="p_lat")
        else:
            direction, score_lat = "Centre", 0

        if st.button("Valider Coup Parcours", type="primary"):
            delta = dist_real - obj_dist if obj_dist > 0 else 0
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': 'Parcours', 'club': club,
                'strat_dist': obj_dist, 'distance': dist_real, 'direction': direction,
                'score_lateral': score_lat, 'real_effet': real_effet, 'type_coup': 'Jeu Long',
                'delta_dist': delta, 'resultat_putt': 'N/A'
            })
            st.success("Coup noté !")

    # --- B. CARTE DE SCORE (FIN DE PARTIE) ---
    with col_carte:
        st.header("📋 Carte de Score")
        with st.form("score_form"):
            st.write("Fin de partie - Résumé")
            c_sc1, c_sc2 = st.columns(2)
            with c_sc1:
                score_f = st.number_input("Score Final (Total)", 60, 120, 80)
                nb_putts = st.number_input("Nb Putts", 18, 50, 32)
            with c_sc2:
                gir = st.number_input("GIR (Greens Touchés)", 0, 18, 5)
                fairways = st.number_input("Fairways Touchés", 0, 14, 5)
            
            if st.form_submit_button("Enregistrer la Partie"):
                st.session_state['parties'].append({
                    'date': str(datetime.date.today()),
                    'score': score_f, 'putts': nb_putts, 'gir': gir, 'fairways': fairways
                })
                st.success("Partie ajoutée à l'historique !")

        # Historique des parties
        if st.session_state['parties']:
            st.subheader("Historique")
            df_part = pd.DataFrame(st.session_state['parties'])
            st.dataframe(df_part.style.highlight_min(subset=['score'], color='lightgreen'), use_container_width=True)

# ==================================================
# ONGLET 2 : PRACTICE LIBRE
# ==================================================
with tab_practice:
    st.header("🚜 Practice Libre (Drill)")
    
    c_pr1, c_pr2, c_pr3 = st.columns(3)
    with c_pr1:
        club_pr = st.selectbox("Club", CLUBS_ORDER, key="pr_club")
        # Plus de type de coup ici
    with c_pr2:
        obj_pr = st.number_input("Cible (m)", 0, 300, DIST_REF.get(club_pr, 100), key="pr_obj")
        dist_pr = st.number_input("Réalisé (m)", 0, 300, int(obj_pr), key="pr_real")
    with c_pr3:
        dir_pr = st.radio("Direction", ["Gauche", "Centre", "Droite"], key="pr_dir")
        lat_pr = st.slider("Dispersion (0-5)", 0, 5, 0, key="pr_lat")

    if st.button("Enregistrer Practice"):
        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': 'Practice', 'club': club_pr,
            'strat_dist': obj_pr, 'distance': dist_pr, 'direction': dir_pr,
            'score_lateral': lat_pr, 'type_coup': 'Jeu Long', 'resultat_putt': 'N/A',
            'delta_dist': dist_pr - obj_pr
        })
        st.success("Ballon noté !")

# ==================================================
# ONGLET 3 : COMBINE TEST (NOUVEAU)
# ==================================================
with tab_combine:
    st.header("🏆 Combine Test Challenge")
    st.markdown("L'IA sélectionne 3 clubs. Vous tapez 5 balles avec chaque club. Objectif : Précision.")

    # 1. INITIALISATION DU TEST
    if st.button("🎲 Lancer un Nouveau Combine"):
        # Choix aléatoire de 3 clubs (hors putter)
        candidates = [c for c in CLUBS_ORDER if c != "Putter"]
        selected_clubs = np.random.choice(candidates, 3, replace=False)
        
        # Définition des cibles
        targets = []
        for c in selected_clubs:
            base = DIST_REF[c]
            # On ajoute une variation aléatoire pour rendre l'exercice difficile
            variation = np.random.randint(-5, 6) 
            targets.append({'club': c, 'target': base + variation})
            
        st.session_state['combine_state'] = {
            'clubs_info': targets,
            'current_club_idx': 0,
            'current_shot': 1,
            'results': [],
            'score_total': 0
        }
        st.rerun()

    # 2. DÉROULEMENT DU TEST
    state = st.session_state['combine_state']
    
    if state:
        current_idx = state['current_club_idx']
        
        if current_idx < 3:
            info = state['clubs_info'][current_idx]
            club_actuel = info['club']
            cible_actuelle = info['target']
            shot_num = state['current_shot']
            
            st.info(f"### Club {current_idx + 1}/3 : {club_actuel} | Cible : {cible_actuelle}m")
            st.progress((current_idx * 5 + shot_num) / 15)
            st.write(f"**Balle {shot_num} / 5**")
            
            # Saisie du coup Combine
            c_comb1, c_comb2 = st.columns(2)
            with c_comb1:
                dist_c = st.number_input("Distance réalisée", 0, 350, cible_actuelle, key=f"c_{current_idx}_{shot_num}")
            with c_comb2:
                lat_c = st.slider("Dispersion (0=Parfait, 5=Raté)", 0, 5, 0, key=f"cl_{current_idx}_{shot_num}")
                dir_c = "Centre" if lat_c == 0 else st.radio("Côté", ["Gauche", "Droite"], horizontal=True, key=f"cd_{current_idx}_{shot_num}")

            if st.button("Valider Balle"):
                # Calcul Points (0-100)
                # Précision distance
                err_dist = abs(dist_c - cible_actuelle)
                pts_dist = max(0, 50 - (err_dist * 2)) # 50 pts max pour distance
                # Précision latérale
                pts_lat = max(0, 50 - (lat_c * 10)) # 50 pts max pour direction
                score_balle = pts_dist + pts_lat
                
                # Sauvegarde dans l'historique global
                st.session_state['coups'].append({
                    'date': str(datetime.date.today()), 'mode': 'Combine', 'club': club_actuel,
                    'strat_dist': cible_actuelle, 'distance': dist_c, 'direction': dir_c,
                    'score_lateral': lat_c, 'type_coup': 'Jeu Long', 'points_test': score_balle,
                    'resultat_putt': 'N/A', 'delta_dist': dist_c - cible_actuelle
                })
                
                state['score_total'] += score_balle
                
                # Incrémentation
                if state['current_shot'] < 5:
                    state['current_shot'] += 1
                else:
                    state['current_shot'] = 1
                    state['current_club_idx'] += 1
                
                st.rerun()
        else:
            # FIN DU TEST
            st.success(f"🎉 Combine Terminé ! Score Final : {int(state['score_total'] / 15)} / 100")
            if st.button("Fermer le Test"):
                st.session_state['combine_state'] = None
                st.rerun()
    else:
        st.write("En attente de lancement...")

# ==================================================
# ONGLET 4 : ANALYSE CLUB (V21)
# ==================================================
with tab_dna:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long']
        
        if not df_long.empty:
            st.header("🧬 Club DNA")
            sel_club = st.selectbox("Club", df_long['club'].unique())
            df_c = df_long[df_long['club'] == sel_club]
            
            col_prac, col_parc = st.columns(2)
            
            # Helper Ellipse
            def plot_ellipse(ax, d, color, title):
                if d.empty: return
                def get_x(r):
                    x = r['score_lateral'] * 5 
                    if r['direction'] == 'Gauche': return -x
                    if r['direction'] == 'Droite': return x
                    return np.random.normal(0,1)
                d['x'] = d.apply(get_x, axis=1)
                ax.scatter(d['x'], d['distance'], c=color, alpha=0.5)
                if len(d) > 3:
                    ell = Ellipse(xy=(d['x'].mean(), d['distance'].mean()), width=d['x'].std()*4, height=d['distance'].std()*4, edgecolor=color, facecolor='none')
                    ax.add_artist(ell)
                ax.set_title(title)

            with col_prac:
                fig1, ax1 = plt.subplots()
                plot_ellipse(ax1, df_c[df_c['mode'] == 'Practice'], 'blue', 'Practice')
                st.pyplot(fig1)
            with col_parc:
                fig2, ax2 = plt.subplots()
                plot_ellipse(ax2, df_c[df_c['mode'] == 'Parcours'], 'red', 'Parcours')
                st.pyplot(fig2)
    else:
        st.info("Pas de données.")

# ==================================================
# ONGLET 5 : BAG MAPPING (V21)
# ==================================================
with tab_sac:
    if st.session_state['coups']:
        st.header("🎒 Étalonnage")
        df = pd.DataFrame(st.session_state['coups'])
        df_sac = df[df['type_coup'] == 'Jeu Long']
        
        if not df_sac.empty:
            df_sac['club'] = pd.Categorical(df_sac['club'], categories=CLUBS_ORDER, ordered=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_sac.sort_values('club'), x='club', y='distance', ax=ax)
            st.pyplot(fig)
            
            stats = df_sac.groupby('club', observed=True)['distance'].agg(['mean', 'std', 'max']).round(1)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

# ==================================================
# ONGLET 6 : PUTTING 360 (V21)
# ==================================================
with tab_putt:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_putt = df[df['type_coup'] == 'Putt']
        
        if not df_putt.empty:
            st.header("🟢 Analyse Putting")
            
            c_p1, c_p2 = st.columns([2, 1])
            
            with c_p1:
                st.subheader("Boussole des Ratés")
                def get_coords(r):
                    res = r['resultat_putt']
                    if res == "Dans le trou": return 0,0
                    x, y = 0, 0
                    if "Gauche" in res: x = -1
                    if "Droite" in res: x = 1
                    if "Court" in res: y = -1
                    if "Long" in res: y = 1
                    return x + np.random.normal(0,0.1), y + np.random.normal(0,0.1)
                
                coords = df_putt.apply(get_coords, axis=1, result_type='expand')
                fig, ax = plt.subplots()
                ax.scatter(coords[0], coords[1], c='purple', s=100, alpha=0.6)
                ax.axhline(0, c='gray'); ax.axvline(0, c='gray')
                ax.set_xlim(-2,2); ax.set_ylim(-2,2)
                ax.set_xlabel("Gauche <-> Droite"); ax.set_ylabel("Court <-> Long")
                st.pyplot(fig)
                
            with c_p2:
                st.subheader("Détail")
                counts = df_putt['resultat_putt'].value_counts()
                st.bar_chart(counts)
                
            st.subheader("Tableau par Distance")
            df_putt['Dist_Zone'] = pd.cut(df_putt['strat_dist'], bins=[0, 2, 5, 10, 30], labels=["0-2m", "2-5m", "5-10m", "+10m"])
            df_putt['Success'] = df_putt['resultat_putt'] == "Dans le trou"
            piv = df_putt.groupby('Dist_Zone', observed=False)['Success'].mean() * 100
            st.dataframe(piv.to_frame("% Réussite").style.background_gradient(cmap="Greens"), use_container_width=True)
