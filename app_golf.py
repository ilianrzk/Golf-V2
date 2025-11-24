import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse, Circle
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 27.0 Practice Master", layout="wide")

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

# Carte de score active
if 'current_card' not in st.session_state:
    st.session_state['current_card'] = pd.DataFrame({
        'Trou': range(1, 19),
        'Par': [4]*18, 'Score': [0]*18, 'Putts': [0]*18
    })
if 'current_hole' not in st.session_state: st.session_state['current_hole'] = 1
if 'shots_on_current_hole' not in st.session_state: st.session_state['shots_on_current_hole'] = 0
if 'putts_on_current_hole' not in st.session_state: st.session_state['putts_on_current_hole'] = 0

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
    "Dans le trou", 
    "Raté - Court", "Raté - Long", 
    "Raté - Gauche", "Raté - Droite",
    "Raté - Court/Gauche", "Raté - Court/Droite",
    "Raté - Long/Gauche", "Raté - Long/Droite"
]

DIST_REF = {
    "Driver": 220, "Bois 5": 200, "Hybride": 180, "Fer 3": 170,
    "Fer 5": 160, "Fer 6": 150, "Fer 7": 140, "Fer 8": 130, "Fer 9": 120,
    "PW": 110, "50°": 100, "55°": 90, "60°": 80, "Putter": 3
}

# --- BARRE LATÉRALE ---
st.sidebar.title("⚙️ Data Lab")

uploaded_file = st.sidebar.file_uploader("📂 Importer CSV", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        current_data = pd.DataFrame(st.session_state['coups'])
        combined = pd.concat([current_data, df_loaded], ignore_index=True).drop_duplicates()
        st.session_state['coups'] = combined.to_dict('records')
        st.sidebar.success(f"{len(df_loaded)} coups importés !")
    except Exception as e:
        st.sidebar.error(f"Erreur : {e}")

st.sidebar.markdown("---")

if st.sidebar.button("Générer Données V27 (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(30)]
    
    for _ in range(400):
        mode = np.random.choice(["Parcours", "Practice", "Combine"], p=[0.6, 0.3, 0.1])
        club = np.random.choice(["Driver", "Fer 7", "Putter"])
        
        base_entry = {
            'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
            'strat_dist': 0, 'distance': 0, 'score_lateral': 0, 'direction': 'Centre',
            'strat_type': 'Entraînement', 'resultat_putt': 'N/A', 'delta_dist': 0, 
            'points_test': 0, 'err_longueur': 'Bonne Longueur', 'lie': 'Fairway',
            'strat_effet': 'N/A', 'real_effet': 'N/A', 'amplitude': 'Plein', 'contact': 'Bon'
        }

        if club == "Putter":
            dist = round(np.random.uniform(0.5, 15.0), 1)
            res = "Dans le trou" if np.random.random() < 0.4 else np.random.choice(PUTT_RESULTS[1:])
            base_entry.update({
                'strat_dist': dist, 'distance': dist, 'resultat_putt': res, 
                'type_coup': 'Putt', 'strat_type': 'Putt', 'lie': 'Green'
            })
        else:
            target = DIST_REF[club]
            real = np.random.normal(target, 10)
            lat = np.random.randint(0, 4)
            direc = "Centre" if lat == 0 else np.random.choice(["Gauche", "Droite"])
            st_eff = np.random.choice(["Tout droit", "Fade", "Draw"])
            
            base_entry.update({
                'strat_dist': target, 'distance': real, 'score_lateral': lat, 'direction': direc,
                'type_coup': 'Jeu Long', 'strat_type': "Attaque de Green", 
                'delta_dist': real-target, 'strat_effet': st_eff
            })
            
        new_data.append(base_entry)
            
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données V27 générées !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v27.csv", "text/csv")

if st.sidebar.button("🗑️ Reset Tout"): 
    st.session_state['coups'] = []
    st.session_state['parties'] = []
    st.session_state['combine_state'] = None
    st.session_state['current_hole'] = 1
    st.session_state['shots_on_current_hole'] = 0
    st.session_state['putts_on_current_hole'] = 0
    st.session_state['current_card']['Score'] = 0
    st.session_state['current_card']['Putts'] = 0

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 27.0 : Practice Master")

tab_parcours, tab_practice, tab_combine, tab_dna, tab_sac, tab_putt = st.tabs([
    "⛳ Parcours & Score", 
    "🚜 Practice Libre",
    "🏆 Combine Test",
    "🧬 Analyse Club", 
    "🎒 Bag Mapping",
    "🟢 Putting 360"
])

# --- HELPER GRAPH ---
def plot_dispersion_analysis(ax, data, title, color):
    if data.empty or len(data) < 2: return
    def get_x(row):
        x = row['score_lateral'] * 5 
        if row['direction'] == 'Gauche': return -x
        if row['direction'] == 'Droite': return x
        return np.random.normal(0, 1)
    data = data.copy()
    data['x_viz'] = data.apply(get_x, axis=1)
    ax.scatter(data['x_viz'], data['distance'], c=color, alpha=0.6, s=60, edgecolors='white')
    target = data['strat_dist'].mean()
    if target > 0: ax.scatter([0], [target], c='green', marker='*', s=150, label='Cible')
    if len(data) > 3:
        try:
            cov = np.cov(data['x_viz'], data['distance'])
            lambda_, v = np.linalg.eig(cov)
            ell = Ellipse(xy=(data['x_viz'].mean(), data['distance'].mean()),
                          width=np.sqrt(lambda_[0])*4, height=np.sqrt(lambda_[1])*4,
                          angle=np.rad2deg(np.arccos(v[0, 0])), 
                          edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
            ax.add_artist(ell)
        except: pass
    ax.set_title(title)
    ax.set_xlabel("Gauche <---> Droite")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)

# ==================================================
# ONGLET 1 : PARCOURS & CARTE
# ==================================================
with tab_parcours:
    col_saisie, col_carte = st.columns([1, 1])
    
    with col_saisie:
        st.header(f"Trou {st.session_state['current_hole']}")
        idx_hole = st.session_state['current_hole'] - 1
        current_par = st.session_state['current_card'].at[idx_hole, 'Par']
        
        st.info(f"Par {current_par} | Coups joués : {st.session_state['shots_on_current_hole']}")

        club = st.selectbox("Club", CLUBS_ORDER, key="p_club")
        
        st.markdown("---")
        c_int, c_real = st.columns(2)
        
        # 1. INTENTION (Avant)
        with c_int:
            st.subheader("🧠 Intention")
            if club == "Putter":
                obj_dist = st.number_input("Dist. Cible (m)", 0.0, 30.0, 1.5, 0.1, format="%.1f", key="p_obj_putt")
                strat_effet = "N/A"
                shot_type = "Putt"
            else:
                shot_type = st.selectbox("Type", SHOT_TYPES, key="p_type")
                if shot_type == "Départ (Tee Shot)" and current_par > 3:
                    obj_dist = 0.0
                    st.caption("🚀 Objectif : Max")
                else:
                    obj_dist = st.number_input("Cible (m)", 0, 350, DIST_REF.get(club, 100), key="p_obj_long")
                
                strat_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade", "Draw", "Balle Basse"], key="p_eff_v")

        # 2. RÉALITÉ (Après)
        with c_real:
            st.subheader("🎯 Réalité")
            if club == "Putter":
                dist_real = st.number_input("Dist. Réelle (m)", 0.0, 30.0, obj_dist, 0.1, format="%.1f", key="p_real_putt")
                res_putt = st.selectbox("Résultat", PUTT_RESULTS, key="p_res_putt")
                direction, score_lat, real_effet = "Centre", 0, "N/A"
            else:
                dist_real = st.number_input("Réalisé (m)", 0, 350, int(obj_dist) if obj_dist>0 else 200, key="p_real_long")
                real_effet = st.selectbox("Effet Réalisé", ["Tout droit", "Fade", "Draw", "Push", "Pull", "Hook", "Slice", "Gratte", "Top"], key="p_eff_r")
                
                c_d1, c_d2 = st.columns(2)
                with c_d1: direction = st.radio("Axe", ["Gauche", "Centre", "Droite"], horizontal=True, key="p_dir")
                with c_d2: score_lat = st.slider("Écart (0-5)", 0, 5, 0, key="p_lat")
                res_putt = "N/A"

        if st.button("Valider Coup (+1 Score)", type="primary"):
            delta = dist_real - obj_dist if obj_dist > 0 else 0
            err_L = "Court" if delta < -5 else ("Long" if delta > 5 else "Bonne Longueur")
            
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': 'Parcours', 'club': club,
                'strat_dist': obj_dist, 'distance': dist_real, 'direction': direction,
                'score_lateral': score_lat, 'real_effet': real_effet, 'strat_effet': strat_effet,
                'type_coup': 'Putt' if club=='Putter' else 'Jeu Long',
                'delta_dist': delta, 'resultat_putt': res_putt, 'err_longueur': err_L,
                'strat_type': shot_type, 'par_trou': current_par
            })
            
            st.session_state['shots_on_current_hole'] += 1
            if club == "Putter":
                st.session_state['putts_on_current_hole'] += 1
                
            idx = st.session_state['current_hole'] - 1
            st.session_state['current_card'].at[idx, 'Score'] = st.session_state['shots_on_current_hole']
            st.session_state['current_card'].at[idx, 'Putts'] = st.session_state['putts_on_current_hole']
            
            if res_putt == "Dans le trou":
                st.balloons()
                st.session_state['shots_on_current_hole'] = 0
                st.session_state['putts_on_current_hole'] = 0
                if st.session_state['current_hole'] < 18:
                    st.session_state['current_hole'] += 1
                st.rerun()
            else:
                st.success("Noté !")

        c_prev, c_next = st.columns(2)
        if c_prev.button("<< Précédent"):
             if st.session_state['current_hole'] > 1:
                st.session_state['current_hole'] -= 1
                st.rerun()
        if c_next.button("Suivant >>"):
            st.session_state['shots_on_current_hole'] = 0
            st.session_state['putts_on_current_hole'] = 0
            if st.session_state['current_hole'] < 18:
                st.session_state['current_hole'] += 1
                st.rerun()

    with col_carte:
        st.header("📋 Carte")
        edited_df = st.data_editor(
            st.session_state['current_card'],
            column_config={
                "Trou": st.column_config.NumberColumn(disabled=True),
                "Par": st.column_config.NumberColumn(min_value=3, max_value=5),
                "Score": st.column_config.NumberColumn(min_value=0),
                "Putts": st.column_config.NumberColumn(min_value=0),
            },
            hide_index=True,
            use_container_width=True,
            height=400
        )
        st.session_state['current_card'] = edited_df
        
        played = edited_df[edited_df['Score'] > 0]
        tot = played['Score'].sum()
        par_tot = played['Par'].sum()
        
        k1, k2 = st.columns(2)
        k1.metric("Total Score", f"{tot}", f"{tot - par_tot:+}" if tot > 0 else "")
        k2.metric("Total Putts", f"{played['Putts'].sum()}")
        
        if st.button("💾 Sauvegarder Partie"):
            st.session_state['parties'].append({
                'date': str(datetime.date.today()), 'score': int(tot), 'putts': int(played['Putts'].sum())
            })
            st.success("Sauvegardé !")

# ==================================================
# ONGLET 2 : PRACTICE (MODIFIÉ)
# ==================================================
with tab_practice:
    st.header("🚜 Practice Libre")
    c_pr1, c_pr2, c_pr3 = st.columns(3)
    
    with c_pr1: 
        club_pr = st.selectbox("Club", CLUBS_ORDER, key="pr_club")
        # AJOUT : Contact
        contact_pr = st.selectbox("Contact", ["Bon", "Gratte", "Top", "Pointe", "Talon"], key="pr_cont")
        
    with c_pr2: 
        obj_pr = st.number_input("Cible (m)", 0, 300, DIST_REF.get(club_pr, 100), key="pr_obj")
        dist_pr = st.number_input("Réalisé (m)", 0, 300, int(obj_pr), key="pr_real")
        # AJOUT : Effet Voulu
        strat_eff_pr = st.selectbox("Effet Voulu", ["Tout droit", "Fade", "Draw", "Balle Basse"], key="pr_eff_v")
        
    with c_pr3:
        dir_pr = st.radio("Direction", ["Gauche", "Centre", "Droite"], key="pr_dir")
        lat_pr = st.slider("Dispersion", 0, 5, 0, key="pr_lat")
        # AJOUT : Effet Réalisé
        real_eff_pr = st.selectbox("Effet Réalisé", ["Tout droit", "Fade", "Draw", "Push", "Pull", "Hook", "Slice"], key="pr_eff_r")

    if st.button("Enregistrer Practice"):
        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': 'Practice', 'club': club_pr,
            'strat_dist': obj_pr, 'distance': dist_pr, 'direction': dir_pr,
            'score_lateral': lat_pr, 'type_coup': 'Jeu Long', 'resultat_putt': 'N/A',
            'delta_dist': dist_pr - obj_pr, 'contact': contact_pr,
            'strat_effet': strat_eff_pr, 'real_effet': real_eff_pr
        })
        st.success("OK")

# ==================================================
# ONGLET 3 : COMBINE
# ==================================================
with tab_combine:
    st.header("🏆 Combine")
    if st.button("🎲 Lancer"):
        cands = [c for c in CLUBS_ORDER if c != "Putter"]
        sels = np.random.choice(cands, 3, replace=False)
        targs = [{'club': c, 'target': DIST_REF[c] + np.random.randint(-5, 6)} for c in sels]
        st.session_state['combine_state'] = {'clubs_info': targs, 'current_club_idx': 0, 'current_shot': 1, 'score_total': 0}
        st.rerun()

    stt = st.session_state['combine_state']
    if stt and stt['current_club_idx'] < 3:
        inf = stt['clubs_info'][stt['current_club_idx']]
        st.info(f"Club : {inf['club']} | Cible : {inf['target']}m | Balle {stt['current_shot']}/5")
        c_c1, c_c2 = st.columns(2)
        with c_c1: dist_c = st.number_input("Distance", 0, 350, inf['target'], key="c_dist")
        with c_c2: lat_c = st.slider("Dispersion", 0, 5, 0, key="c_lat")
        if st.button("Valider"):
            pts = max(0, 50 - (abs(dist_c - inf['target'])*2)) + max(0, 50 - (lat_c*10))
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': 'Combine', 'club': inf['club'],
                'strat_dist': inf['target'], 'distance': dist_c, 'score_lateral': lat_c, 
                'direction': 'Centre' if lat_c==0 else 'Gauche',
                'type_coup': 'Jeu Long', 'points_test': pts, 'resultat_putt': 'N/A'
            })
            stt['score_total'] += pts
            if stt['current_shot'] < 5: stt['current_shot'] += 1
            else: 
                stt['current_shot'] = 1
                stt['current_club_idx'] += 1
            st.rerun()
    elif stt:
        st.success(f"Score Final : {int(stt['score_total']/15)}/100")
        if st.button("Fermer"):
            st.session_state['combine_state'] = None
            st.rerun()
            
    st.markdown("---")
    st.subheader("📊 Stats Combine")
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_c = df[df['mode'] == 'Combine']
        if not df_c.empty:
            # SCORE SUR 100
            score_avg = df_c['points_test'].mean()
            st.metric("Score Moyen Combine", f"{score_avg:.0f} / 100")
            
            sl = st.selectbox("Club Combine", df_c['club'].unique())
            subset = df_c[df_c['club'] == sl]
            
            c_a1, c_a2 = st.columns(2)
            with c_a1:
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, subset, "Dispersion", "orange")
                st.pyplot(fig)
            with c_a2:
                st.metric("Prof. ±m", f"± {subset['distance'].std():.1f}m")
                st.metric("Latéral /5", f"{subset['score_lateral'].mean():.1f}/5")

# ==================================================
# ONGLET 4 : ANALYSE CLUB
# ==================================================
with tab_dna:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_l = df[df['type_coup'] == 'Jeu Long']
        if not df_l.empty:
            sel = st.selectbox("Club", df_l['club'].unique())
            sub = df_l[df_l['club'] == sel]
            
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, sub[sub['mode']=='Practice'], "Practice", "blue")
                st.pyplot(fig)
                if not sub[sub['mode']=='Practice'].empty:
                    st.metric("Prof.", f"± {sub[sub['mode']=='Practice']['distance'].std():.1f}m")
                    st.metric("Lat.", f"{sub[sub['mode']=='Practice']['score_lateral'].mean():.1f}/5")
            with c2:
                fig, ax = plt.subplots()
                plot_dispersion_analysis(ax, sub[sub['mode']=='Parcours'], "Parcours", "red")
                st.pyplot(fig)
                if not sub[sub['mode']=='Parcours'].empty:
                    st.metric("Prof.", f"± {sub[sub['mode']=='Parcours']['distance'].std():.1f}m")
                    st.metric("Lat.", f"{sub[sub['mode']=='Parcours']['score_lateral'].mean():.1f}/5")
            
            st.subheader("Maîtrise des Effets")
            df_eff = sub[sub['strat_effet'].isin(["Fade", "Draw", "Tout droit"])]
            if not df_eff.empty:
                df_eff['OK'] = df_eff.apply(lambda x: 1 if x['strat_effet'] in x['real_effet'] else 0, axis=1)
                res = df_eff.groupby('strat_effet')['OK'].mean() * 100
                st.dataframe(res.to_frame("% Réussite").style.background_gradient(cmap="Greens"), use_container_width=True)

# ==================================================
# ONGLET 5 : BAG MAPPING (COULEURS)
# ==================================================
with tab_sac:
    if st.session_state['coups']:
        st.header("🎒 Étalonnage")
        df = pd.DataFrame(st.session_state['coups'])
        df_s = df[df['type_coup'] == 'Jeu Long']
        
        if not df_s.empty:
            df_s['club'] = pd.Categorical(df_s['club'], categories=CLUBS_ORDER, ordered=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='club', y='distance', data=df_s, ax=ax)
            st.pyplot(fig)
            
            stats = df_s.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'max', 'std']).round(1)
            stats.columns = ['Nb Coups', 'Moyenne (m)', 'Max (m)', 'Écart Type (m)']
            
            # COULEURS SÉMANTIQUES
            st.dataframe(
                stats.style.background_gradient(cmap="Blues", subset=['Moyenne (m)', 'Max (m)'])
                           .background_gradient(cmap="RdYlGn_r", subset=['Écart Type (m)']), 
                use_container_width=True
            )

# ==================================================
# ONGLET 6 : PUTTING (TABLEAU + COULEURS)
# ==================================================
with tab_putt:
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_p = df[df['type_coup'] == 'Putt']
        
        if not df_p.empty:
            st.header("🟢 Analyse Putting")
            
            # STATS PAR ZONE (COULEURS)
            st.subheader("📊 Réussite par Zone")
            df_p['Zone'] = pd.cut(df_p['strat_dist'], bins=[0, 2, 5, 10, 30], labels=["0-2m", "2-5m", "5-10m", "+10m"])
            df_p['Success'] = df_p['resultat_putt'] == "Dans le trou"
            
            piv = df_p.groupby('Zone', observed=False).agg(
                Tentatives=('resultat_putt', 'count'),
                Reussites=('Success', 'sum')
            )
            piv['% Réussite'] = (piv['Reussites'] / piv['Tentatives'] * 100).round(1)
            # COULEUR : Vert = Bon %
            st.dataframe(piv.style.background_gradient(cmap="RdYlGn", subset=['% Réussite']), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Boussole")
                def get_pc(r):
                    res = r['resultat_putt']
                    if "Dans" in res: return 0,0
                    x, y = 0, 0
                    if "Gauche" in res: x = -1
                    if "Droite" in res: x = 1
                    if "Court" in res: y = -1
                    if "Long" in res: y = 1
                    return x + np.random.normal(0,0.1), y + np.random.normal(0,0.1)
                
                coords = df_p.apply(get_pc, axis=1, result_type='expand')
                fig, ax = plt.subplots()
                ax.scatter(coords[0], coords[1], alpha=0.5, s=100, c='purple')
                ax.axhline(0, c='gray'); ax.axvline(0, c='gray')
                st.pyplot(fig)
            
            with c2:
                st.subheader("Causes des Ratés")
                misses = df_p[df_p['resultat_putt'] != "Dans le trou"]
                if not misses.empty:
                    cnt = misses['resultat_putt'].value_counts()
                    st.bar_chart(cnt)
                else:
                    st.success("Aucun raté !")
