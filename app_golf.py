import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 23.0 Scorecard Pro", layout="wide")

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
    st.session_state['combine_state'] = None

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

# --- BARRE LATÉRALE : DATA LAB ---
st.sidebar.title("⚙️ Data Lab")

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

if st.sidebar.button("Générer Données V23 (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(30)]
    for _ in range(300):
        mode = np.random.choice(["Parcours", "Practice", "Combine"], p=[0.5, 0.3, 0.2])
        club = np.random.choice(["Driver", "Fer 7", "Putter"])
        
        if club == "Putter":
            res = np.random.choice(PUTT_RESULTS)
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
    
    # Génération d'une partie complète pour tester la carte de score
    fake_card = []
    for i in range(1, 19):
        fake_card.append({'Trou': i, 'Par': 4, 'Score': np.random.randint(3, 7), 'Putts': 2})
    st.session_state['parties'].append({
        'date': str(datetime.date.today()), 'detail': fake_card, 'total_score': sum(x['Score'] for x in fake_card)
    })
    
    st.sidebar.success("Données générées !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v23.csv", "text/csv")

if st.sidebar.button("🗑️ Reset Tout"): 
    st.session_state['coups'] = []
    st.session_state['parties'] = []
    st.session_state['combine_state'] = None

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 23.0 : Scorecard Pro & Combine")

tab_parcours, tab_practice, tab_combine, tab_dna, tab_sac, tab_putt = st.tabs([
    "⛳ Parcours & Score", 
    "🚜 Practice Libre",
    "🏆 Combine Test",
    "🧬 Analyse Club", 
    "🎒 Bag Mapping",
    "🟢 Putting 360"
])

# --- HELPER GRAPH DISPERSION ---
def plot_dispersion_analysis(ax, data, title, color):
    """Affiche la dispersion avec ellipse et cibles"""
    if data.empty: return
    
    def get_x(row):
        x = row['score_lateral'] * 5 # Spatialisation visuelle pour le graph
        if row['direction'] == 'Gauche': return -x
        if row['direction'] == 'Droite': return x
        return np.random.normal(0, 1)
    
    data = data.copy()
    data['x_viz'] = data.apply(get_x, axis=1)
    
    ax.scatter(data['x_viz'], data['distance'], c=color, alpha=0.6, s=60, edgecolors='white')
    target = data['strat_dist'].mean()
    if target > 0: ax.scatter([0], [target], c='green', marker='*', s=150, label='Cible Moyenne')
    
    if len(data) > 3:
        cov = np.cov(data['x_viz'], data['distance'])
        lambda_, v = np.linalg.eig(cov)
        ell = Ellipse(xy=(data['x_viz'].mean(), data['distance'].mean()),
                      width=np.sqrt(lambda_[0])*4, height=np.sqrt(lambda_[1])*4,
                      angle=np.rad2deg(np.arccos(v[0, 0])), 
                      edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
        ax.add_artist(ell)
    
    ax.set_title(title)
    ax.set_xlabel("Gauche <---> Droite")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)

# ==================================================
# ONGLET 1 : PARCOURS & VRAIE CARTE DE SCORE
# ==================================================
with tab_parcours:
    col_saisie, col_carte = st.columns([1, 1])
    
    # --- A. SAISIE COUP PARCOURS ---
    with col_saisie:
        st.header("📝 Coup par Coup")
        club = st.selectbox("Club", CLUBS_ORDER, key="p_club")
        par_trou = st.selectbox("Par du trou", [3, 4, 5], key="p_par")
        
        c1, c2 = st.columns(2)
        with c1:
            shot_type = st.selectbox("Type", SHOT_TYPES, key="p_type")
            if shot_type == "Départ (Tee Shot)" and par_trou > 3:
                obj_dist = 0.0
                st.caption("🎯 Objectif : Max")
            else:
                obj_dist = st.number_input("Cible (m)", 0, 350, DIST_REF.get(club, 100), key="p_obj")
                
        with c2:
            dist_real = st.number_input("Réalisé (m)", 0, 350, int(obj_dist) if obj_dist>0 else 200, key="p_real")
            real_effet = st.selectbox("Effet", ["Tout droit", "Fade", "Draw", "Raté"], key="p_eff")

        if club != "Putter":
            cc1, cc2 = st.columns(2)
            with cc1: direction = st.radio("Axe", ["Gauche", "Centre", "Droite"], horizontal=True, key="p_dir")
            with cc2: score_lat = st.slider("Écart (0-5)", 0, 5, 0, key="p_lat")
        else:
            direction, score_lat = "Centre", 0

        if st.button("Valider Coup", type="primary"):
            delta = dist_real - obj_dist if obj_dist > 0 else 0
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': 'Parcours', 'club': club,
                'strat_dist': obj_dist, 'distance': dist_real, 'direction': direction,
                'score_lateral': score_lat, 'real_effet': real_effet, 'type_coup': 'Jeu Long',
                'delta_dist': delta, 'resultat_putt': 'N/A'
            })
            st.success("Noté !")

    # --- B. CARTE DE SCORE 18 TROUS (NOUVEAU) ---
    with col_carte:
        st.header("📋 Carte de Score")
        
        # Initialisation d'une carte vierge si besoin
        if 'current_card' not in st.session_state:
            st.session_state['current_card'] = pd.DataFrame({
                'Trou': range(1, 19),
                'Par': [4,4,3,4,5,4,3,4,5, 4,4,3,4,5,4,3,4,5], # Exemple générique
                'Score': [0]*18,
                'Putts': [0]*18
            })

        st.markdown("##### Remplissez votre partie :")
        # Editeur de données interactif
        edited_df = st.data_editor(
            st.session_state['current_card'],
            column_config={
                "Trou": st.column_config.NumberColumn(disabled=True),
                "Par": st.column_config.NumberColumn(min_value=3, max_value=5),
                "Score": st.column_config.NumberColumn(min_value=1, max_value=15),
                "Putts": st.column_config.NumberColumn(min_value=0, max_value=5),
            },
            hide_index=True,
            use_container_width=True,
            height=300
        )
        
        # Calcul des totaux en temps réel
        total_par = edited_df['Par'].sum()
        # On ne compte que les trous joués (score > 0) pour le total actuel
        holes_played = edited_df[edited_df['Score'] > 0]
        total_score = holes_played['Score'].sum()
        total_putts = holes_played['Putts'].sum()
        score_rel = total_score - holes_played['Par'].sum()
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Score Total", f"{total_score}", f"{score_rel:+}" if total_score > 0 else "")
        k2.metric("Total Putts", f"{total_putts}")
        k3.metric("Trous Joués", f"{len(holes_played)}/18")
        
        if st.button("💾 Archiver la Partie (Fin 18 trous)"):
            if len(holes_played) > 0:
                st.session_state['parties'].append({
                    'date': str(datetime.date.today()),
                    'detail': edited_df.to_dict('records'),
                    'total_score': int(total_score),
                    'total_putts': int(total_putts)
                })
                # Reset carte
                st.session_state['current_card']['Score'] = 0
                st.session_state['current_card']['Putts'] = 0
                st.success("Partie archivée dans l'historique !")
            else:
                st.error("Remplissez au moins un score.")

        # Petit historique rapide
        if st.session_state['parties']:
            with st.expander("Voir historique des parties"):
                hist_df = pd.DataFrame(st.session_state['parties'])
                st.dataframe(hist_df[['date', 'total_score', 'total_putts']])

# ==================================================
# ONGLET 2 : PRACTICE
# ==================================================
with tab_practice:
    st.header("🚜 Practice Libre")
    c_pr1, c_pr2, c_pr3 = st.columns(3)
    with c_pr1: club_pr = st.selectbox("Club", CLUBS_ORDER, key="pr_club")
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
# ONGLET 3 : COMBINE TEST (AVEC ANALYSE DÉDIÉE)
# ==================================================
with tab_combine:
    st.header("🏆 Combine Test Challenge")
    
    # 1. Interface de jeu (inchangée V22)
    if st.button("🎲 Lancer un Nouveau Combine"):
        candidates = [c for c in CLUBS_ORDER if c != "Putter"]
        selected_clubs = np.random.choice(candidates, 3, replace=False)
        targets = []
        for c in selected_clubs:
            base = DIST_REF[c]
            targets.append({'club': c, 'target': base + np.random.randint(-5, 6)})
        st.session_state['combine_state'] = {'clubs_info': targets, 'current_club_idx': 0, 'current_shot': 1, 'score_total': 0}
        st.rerun()

    state = st.session_state['combine_state']
    if state:
        if state['current_club_idx'] < 3:
            info = state['clubs_info'][state['current_club_idx']]
            st.info(f"### Club : {info['club']} | Cible : {info['target']}m | Balle {state['current_shot']}/5")
            
            c_c1, c_c2 = st.columns(2)
            with c_c1: dist_c = st.number_input("Distance", 0, 350, info['target'], key=f"c_{state['current_club_idx']}_{state['current_shot']}")
            with c_c2: 
                lat_c = st.slider("Dispersion (0-5)", 0, 5, 0, key=f"cl_comb")
                dir_c = "Centre" if lat_c == 0 else st.radio("Côté", ["Gauche", "Droite"], horizontal=True, key=f"cd_comb")

            if st.button("Valider Balle"):
                err_dist = abs(dist_c - info['target'])
                pts = max(0, 50 - (err_dist*2)) + max(0, 50 - (lat_c*10))
                st.session_state['coups'].append({
                    'date': str(datetime.date.today()), 'mode': 'Combine', 'club': info['club'],
                    'strat_dist': info['target'], 'distance': dist_c, 'direction': dir_c,
                    'score_lateral': lat_c, 'type_coup': 'Jeu Long', 'points_test': pts,
                    'resultat_putt': 'N/A', 'delta_dist': dist_c - info['target']
                })
                state['score_total'] += pts
                if state['current_shot'] < 5: state['current_shot'] += 1
                else: 
                    state['current_shot'] = 1
                    state['current_club_idx'] += 1
                st.rerun()
        else:
            st.success(f"🎉 Score Final : {int(state['score_total'] / 15)} / 100")
            if st.button("Fermer"):
                st.session_state['combine_state'] = None
                st.rerun()

    st.markdown("---")
    
    # 2. ANALYSE SPÉCIALE COMBINE (NOUVEAU DEMANDÉ)
    st.subheader("📊 Analyse des Performances 'Combine'")
    if st.session_state['coups']:
        df_all = pd.DataFrame(st.session_state['coups'])
        df_combine = df_all[df_all['mode'] == 'Combine']
        
        if not df_combine.empty:
            sel_comb = st.selectbox("Voir Club du Combine", df_combine['club'].unique())
            df_c_comb = df_combine[df_combine['club'] == sel_comb]
            
            if not df_c_comb.empty:
                col_ca1, col_ca2 = st.columns(2)
                
                with col_ca1:
                    fig_comb, ax_comb = plt.subplots(figsize=(5, 5))
                    plot_dispersion_analysis(ax_comb, df_c_comb, f"Dispersion Combine : {sel_comb}", "orange")
                    st.pyplot(fig_comb)
                
                with col_ca2:
                    # METRIQUES SPÉCIFIQUES DEMANDÉES
                    st.markdown("#### Indicateurs de Pression")
                    avg_lat = df_c_comb['score_lateral'].mean()
                    std_depth = df_c_comb['distance'].std()
                    avg_score = df_c_comb['points_test'].mean()
                    
                    st.metric("Score Moyen / 100", f"{avg_score:.0f}")
                    st.metric("Dispersion Profondeur", f"± {std_depth:.1f} m")
                    st.metric("Dispersion Latérale (0-5)", f"{avg_lat:.1f} / 5")
            else:
                st.info("Sélectionnez un club joué en mode Combine.")
        else:
            st.warning("Aucune donnée de Combine enregistrée pour l'instant.")

# ==================================================
# ONGLET 4 : ANALYSE CLUB (V20 + Métriques)
# ==================================================
with tab_dna:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_long = df[df['type_coup'] == 'Jeu Long']
        
        if not df_long.empty:
            st.header("🧬 Club DNA (Global)")
            sel_club = st.selectbox("Choisir Club", df_long['club'].unique())
            df_c = df_long[df_long['club'] == sel_club]
            
            col_prac, col_parc = st.columns(2)
            df_p1 = df_c[df_c['mode'] == 'Practice']
            df_p2 = df_c[df_c['mode'] == 'Parcours']
            
            with col_prac:
                fig1, ax1 = plt.subplots()
                plot_dispersion_analysis(ax1, df_p1, "Practice", "blue")
                st.pyplot(fig1)
                if not df_p1.empty:
                    st.metric("Dispersion Profondeur", f"± {df_p1['distance'].std():.1f} m")
                    st.metric("Dispersion Latérale", f"{df_p1['score_lateral'].mean():.1f} / 5")

            with col_parc:
                fig2, ax2 = plt.subplots()
                plot_dispersion_analysis(ax2, df_p2, "Parcours", "red")
                st.pyplot(fig2)
                if not df_p2.empty:
                    st.metric("Dispersion Profondeur", f"± {df_p2['distance'].std():.1f} m")
                    st.metric("Dispersion Latérale", f"{df_p2['score_lateral'].mean():.1f} / 5")
    else:
        st.info("Pas de données.")

# ==================================================
# ONGLET 5 : BAG MAPPING (V20)
# ==================================================
with tab_sac:
    if st.session_state['coups']:
        st.header("🎒 Étalonnage")
        df = pd.DataFrame(st.session_state['coups'])
        df_sac = df[df['type_coup'] == 'Jeu Long']
        if not df_sac.empty:
            df_sac['club'] = pd.Categorical(df_sac['club'], categories=CLUBS_ORDER, ordered=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='club', y='distance', data=df_sac, ax=ax, palette="viridis")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            stats = df_sac.groupby('club', observed=True)['distance'].agg(['mean', 'std', 'max']).round(1)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

# ==================================================
# ONGLET 6 : PUTTING 360 (V20)
# ==================================================
with tab_putt:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_putt = df[df['type_coup'] == 'Putt']
        
        if not df_putt.empty:
            st.header("🟢 Analyse Putting")
            c_p1, c_p2 = st.columns(2)
            
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
                st.subheader("Stats par Distance")
                df_putt['Zone'] = pd.cut(df_putt['strat_dist'], bins=[0, 2, 5, 10, 30], labels=["0-2m", "2-5m", "5-10m", "+10m"])
                df_putt['Success'] = df_putt['resultat_putt'] == "Dans le trou"
                piv = df_putt.groupby('Zone', observed=False)['Success'].mean() * 100
                st.dataframe(piv.to_frame("% Réussite").style.background_gradient(cmap="Greens"), use_container_width=True)
