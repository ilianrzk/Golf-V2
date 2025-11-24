import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GolfShot 13.0 Mastery", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    h4 {color: #4CAF50;}
</style>
""", unsafe_allow_html=True)

# --- DONNÉES & CONSTANTES ---
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

# --- GÉNÉRATEUR V13 ---
st.sidebar.title("⚙️ Data Lab V13")

if st.sidebar.button("Générer Dataset 'Mastery'"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(60)]
    
    for _ in range(400):
        club = np.random.choice(["Driver", "Fer 7", "PW", "55°"])
        mode = np.random.choice(["Practice", "Parcours"], p=[0.3, 0.7])
        
        # Amplitude
        if club in ["PW", "55°"]: ampli = np.random.choice(["Plein", "3/4", "1/2"], p=[0.5, 0.3, 0.2])
        else: ampli = "Plein"

        # Lie
        if mode == "Practice": lie = "Tapis/Tee"
        else:
            if club == "Driver": lie = "Tee"
            else: lie = np.random.choice(["Fairway", "Rough", "Bunker"], p=[0.6, 0.3, 0.1])

        # Intention (Effet)
        strat_effet = np.random.choice(["Tout droit", "Fade", "Draw"], p=[0.7, 0.15, 0.15])

        # Calcul Distance
        dist_target = DIST_REF[club]
        if ampli == "3/4": dist_target *= 0.85
        if ampli == "1/2": dist_target *= 0.60
        
        # Physique du coup
        penalty = 0.88 if lie == "Rough" else (0.75 if lie == "Bunker" else 1.0)
        std_dev = 6 if mode == "Practice" else 12
        dist_real = np.random.normal(dist_target * penalty, std_dev)
        
        # Erreurs
        delta = dist_real - dist_target
        if delta < -6: err_L = "Court"
        elif delta > 6: err_L = "Long"
        else: err_L = "Bonne Longueur"
        
        lat_score = abs(np.random.normal(0, 2))
        lat_score = min(5, int(lat_score))
        direction = "Centre" if lat_score == 0 else np.random.choice(["Gauche", "Droite"])
        
        contact = np.random.choice(["Bon", "Gratte", "Top", "Pointe"], p=[0.65, 0.15, 0.1, 0.1])

        new_data.append({
            'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
            'strat_dist': int(dist_target), 'strat_effet': strat_effet, 'amplitude': ampli, # Intention
            'distance': round(dist_real, 1), 'lie': lie, 'contact': contact,
            'direction': direction, 'score_lateral': lat_score,
            'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Jeu Long'
        })
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("400 Coups générés !")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("📥 Export CSV", convert_df(df_ex), "golf_v13.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE ---
st.title("🏌️‍♂️ GolfShot 13.0 : Mastery")

tab_saisie, tab_sac, tab_dna = st.tabs(["📝 Saisie Complète", "🎒 Bag Mapping & Gapping", "🧬 Analyse Club & Lie"])

# --- ONGLET 1 : SAISIE ---
with tab_saisie:
    col_m, col_c = st.columns(2)
    with col_m: mode = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
    with col_c: club = st.selectbox("Club en main", CLUBS_ORDER)
    
    st.markdown("---")
    col_int, col_real = st.columns(2)

    with col_int:
        st.subheader("1️⃣ INTENTION")
        if club == "Putter":
            obj_dist = st.number_input("Dist. Cible", 0.0, 30.0, 3.0)
            strat_effet = "Aucun"
            amplitude = "Plein"
        else:
            obj_dist = st.number_input("Dist. Cible", 0, 350, DIST_REF.get(club, 100))
            # RETOUR DE L'EFFET SOUHAITÉ
            strat_effet = st.selectbox("Effet Voulu", ["Tout droit", "Fade (G-D)", "Draw (D-G)", "Balle basse"])
            amplitude = st.radio("Amplitude", ["Plein", "3/4", "1/2"], horizontal=True)

    with col_real:
        st.subheader("2️⃣ RÉALITÉ")
        if club == "Putter":
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté"])
            dist_real = obj_dist if res_putt == "Dans le trou" else st.number_input("Dist. Réelle", 0.0, 30.0, obj_dist)
            lie = "Green"
            direction = "Centre"
            score_lat = 0
            contact = "Bon"
        else:
            dist_real = st.number_input("Distance Réelle (m)", 0, 350, int(obj_dist))
            lie = st.selectbox("Situation (Lie)", ["Tee", "Fairway", "Rough", "Bunker"])
            contact = st.selectbox("Contact", ["Bon", "Gratte", "Top", "Pointe", "Talon"])
            
            c_dir, c_sco = st.columns(2)
            with c_dir: direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            with c_sco: score_lat = st.number_input("Écart (0-5)", 0, 5, 0) if direction != "Centre" else 0

    st.markdown("---")
    if st.button("💾 Enregistrer", type="primary", use_container_width=True):
        delta = dist_real - obj_dist
        if delta < -5: err_L = "Court"
        elif delta > 5: err_L = "Long"
        else: err_L = "Bonne Longueur"

        st.session_state['coups'].append({
            'date': str(datetime.date.today()), 'mode': mode.split()[0], 'club': club,
            'strat_dist': obj_dist, 'amplitude': amplitude, 'strat_effet': strat_effet,
            'distance': dist_real, 'lie': lie, 'contact': contact,
            'direction': direction, 'score_lateral': score_lat,
            'delta_dist': delta, 'err_longueur': err_L,
            'type_coup': 'Putt' if club == "Putter" else 'Jeu Long'
        })
        st.success("Coup enregistré !")

# --- DATA PREP ---
if st.session_state['coups']:
    df = pd.DataFrame(st.session_state['coups'])
    df_long = df[df['type_coup'] == 'Jeu Long']
else:
    df_long = pd.DataFrame()

# --- ONGLET 2 : BAG MAPPING (Gapping) ---
with tab_sac:
    if not df_long.empty:
        st.header("🎒 Étalonnage du Sac (Gapping)")
        
        # Filtre Amplitude pour le mapping
        f_ampli_sac = st.selectbox("Amplitude pour le Mapping", ["Plein", "3/4", "1/2"], index=0)
        df_sac = df_long[df_long['amplitude'] == f_ampli_sac].copy()
        
        if not df_sac.empty:
            # Ordre Clubs
            df_sac['club'] = pd.Categorical(df_sac['club'], categories=CLUBS_ORDER, ordered=True)
            df_sac = df_sac.sort_values('club')
            
            # --- 1. GRAPHIQUE VISUEL ---
            fig_bag, ax_bag = plt.subplots(figsize=(12, 5))
            sns.boxplot(x='club', y='distance', data=df_sac, ax=ax_bag, palette="viridis")
            sns.stripplot(x='club', y='distance', data=df_sac, color=".25", size=3, alpha=0.5, ax=ax_bag)
            ax_bag.grid(True, axis='y', alpha=0.3)
            ax_bag.set_title(f"Portées de balles : {f_ampli_sac}")
            st.pyplot(fig_bag)
            
            st.markdown("---")
            
            # --- 2. TABLEAU DE DONNÉES PRÉCISES (AJOUT MAJEUR) ---
            st.subheader("🔢 Données Chiffrées par Club")
            
            # Calculs Stats
            stats_sac = df_sac.groupby('club', observed=True)['distance'].agg(['count', 'mean', 'min', 'max', 'std']).round(1)
            
            # Calcul Efficacité (Moyenne / Max)
            stats_sac['Efficacité (%)'] = (stats_sac['mean'] / stats_sac['max'] * 100).round(0)
            
            # Renommage
            stats_sac.columns = ['Nb Coups', 'Moyenne (m)', 'Min', 'Max', 'Dispersion (±m)', 'Efficacité (%)']
            
            # Affichage "Pro" avec couleurs
            st.dataframe(
                stats_sac.style.background_gradient(cmap="Blues", subset=['Moyenne (m)'])
                               .background_gradient(cmap="RdYlGn", subset=['Efficacité (%)'])
                               .format("{:.1f}", subset=['Moyenne (m)', 'Min', 'Max', 'Dispersion (±m)'])
                               .format("{:.0f}%", subset=['Efficacité (%)']),
                use_container_width=True
            )
            st.info("💡 **Indice de Progression :** Regardez la colonne **'Efficacité'**. Si vous êtes en dessous de 85%, cela signifie que vous avez la puissance (Max) mais pas la maîtrise (Moyenne).")
            
        else:
            st.warning("Pas de données pour cette amplitude.")
    else:
        st.info("En attente de données...")

# --- ONGLET 3 : CLUB DNA (LIE & STATS) ---
with tab_dna:
    if not df_long.empty:
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1: selected_club = st.selectbox("Club à analyser", df_long['club'].unique())
        with col_sel2: selected_ampli = st.selectbox("Amplitude", ["Plein", "3/4", "1/2", "Tout"], index=0)

        df_club = df_long[df_long['club'] == selected_club]
        if selected_ampli != "Tout":
            df_club = df_club[df_club['amplitude'] == selected_ampli]
        
        if not df_club.empty:
            
            # KPI RAPIDES
            st.markdown(f"#### 🔎 Focus : {selected_club}")
            k1, k2, k3, k4 = st.columns(4)
            avg = df_club['distance'].mean()
            k1.metric("Moyenne", f"{avg:.1f}m")
            k2.metric("Précision (Dispersion)", f"± {df_club['distance'].std():.1f}m")
            k3.metric("Meilleur Coup", f"{df_club['distance'].max():.1f}m")
            
            # Calcul Réussite Intention
            reussite_dist = len(df_club[abs(df_club['delta_dist']) < 8]) / len(df_club) * 100
            k4.metric("Précision Cible", f"{reussite_dist:.0f}%", help="% de balles à +/- 8m de la cible")
            
            st.markdown("---")
            
            col_L, col_R = st.columns([1, 1])
            
            # --- SECTION GAUCHE : IMPACT DU LIE (DATA + GRAPH) ---
            with col_L:
                st.subheader("🌿 Impact du Lie (Terrain)")
                
                # 1. Graphique
                if len(df_club['lie'].unique()) > 1:
                    fig_lie, ax_lie = plt.subplots(figsize=(6, 3))
                    sns.boxplot(x='lie', y='distance', data=df_club, ax=ax_lie, palette="Set2")
                    st.pyplot(fig_lie)
                
                # 2. Tableau Data (NOUVEAU)
                st.write("**Pertes de distance par Lie**")
                pivot_lie = df_club.groupby('lie')['distance'].agg(['mean', 'count']).round(1)
                
                # Calcul % par rapport au Tee ou Fairway (Reference)
                ref_dist = pivot_lie['mean'].max() # On prend la meilleure situation comme ref
                pivot_lie['% du Max'] = (pivot_lie['mean'] / ref_dist * 100).round(0)
                pivot_lie['Perte (m)'] = (ref_dist - pivot_lie['mean']).round(1)
                
                pivot_lie.columns = ['Moyenne', 'Nb Coups', '% Performance', 'Perte (m)']
                
                st.dataframe(pivot_lie.style.format("{:.1f}m", subset=['Moyenne', 'Perte (m)']).format("{:.0f}%", subset=['% Performance']))
                st.caption("Ce tableau vous dit exactement combien de mètres retirer quand vous êtes dans le Rough.")

            # --- SECTION DROITE : DISPERSION & CONTACT ---
            with col_R:
                st.subheader("🎯 Qualité & Dispersion")
                
                # Graphique Dispersion
                fig_scat, ax_scat = plt.subplots(figsize=(6, 4))
                
                def get_x(row):
                    x = row['score_lateral'] * 5 
                    if row['direction'] == 'Gauche': return -x
                    if row['direction'] == 'Droite': return x
                    return 0 + np.random.normal(0,1)
                
                df_club['x_viz'] = df_club.apply(get_x, axis=1)
                sns.scatterplot(data=df_club, x='x_viz', y='distance', hue='contact', style='mode', s=100, ax=ax_scat)
                
                # Ellipse
                if len(df_club) > 3:
                    cov = np.cov(df_club['x_viz'], df_club['distance'])
                    lambda_, v = np.linalg.eig(cov)
                    lambda_ = np.sqrt(lambda_)
                    ell = Ellipse(xy=(df_club['x_viz'].mean(), df_club['distance'].mean()),
                                  width=lambda_[0]*4, height=lambda_[1]*4,
                                  angle=np.rad2deg(np.arccos(v[0, 0])), 
                                  edgecolor='red', facecolor='none', linestyle='--')
                    ax_scat.add_artist(ell)
                
                st.pyplot(fig_scat)
                

[Image of box plot chart explanation]

                st.caption("L'ellipse rouge montre votre zone de dispersion principale (95%).")
