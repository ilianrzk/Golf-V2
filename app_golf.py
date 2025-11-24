import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot Data 5.0", layout="wide")

# --- STYLES CSS POUR TABLEAUX ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

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

# --- SIDEBAR : OPTIONS & DATA ---
st.sidebar.title("🧮 Data Center")

# Générateur Avancé
if st.sidebar.button("Générer Dataset Pro (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(10)]
    
    for _ in range(200): # 200 coups simulés
        club = np.random.choice(CLUBS_ORDER)
        d = np.random.choice(dates)
        
        if club == 'Putter':
            dist = np.random.exponential(3) # Beaucoup de putts courts, peu de longs
            if dist < 0.5: dist = 0.5
            # Probabilité de réussite diminue avec la distance
            success_prob = max(0.05, 1 - (dist / 8)) 
            res = "Dans le trou" if np.random.random() < success_prob else "Raté"
            
            new_data.append({
                'date': str(d), 'mode': 'Parcours', 'club': 'Putter',
                'distance': round(dist, 1), 'resultat': res, 'type_coup': 'Putt',
                'pente': np.random.choice(["Plat", "G-D", "D-G"]), 
                'denivele': "Plat", 'green_touche': False
            })
        else:
            # Simulation Fers/Bois
            base = 220 if club=='Driver' else (140 if 'Fer' in club else 90)
            dist_real = np.random.normal(base, 15)
            green_hit = True if (abs(dist_real - base) < 10) and ('Fer' in club or 'PW' in club) else False
            
            new_data.append({
                'date': str(d), 'mode': 'Parcours', 'club': club,
                'distance': round(dist_real, 1), 
                'direction': np.random.choice(["Centre", "Gauche", "Droite"], p=[0.5, 0.25, 0.25]),
                'longueur': np.random.choice(["Ok", "Court", "Long"], p=[0.6, 0.2, 0.2]),
                'resultat': "Jeu", 'type_coup': 'Jeu Long',
                'green_touche': green_hit # Nouveau champ KPI
            })
            
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Dataset chargé (200 coups) !")

# Gestion Fichiers
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Importer CSV", type="csv")
if uploaded_file:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.session_state['coups'] = df_loaded.to_dict('records')
        st.sidebar.success("Données importées")
    except: st.sidebar.error("Erreur CSV")

if st.session_state['coups']:
    df_ex = pd.DataFrame(st.session_state['coups'])
    st.sidebar.download_button("Exporter CSV pour Excel", convert_df(df_ex), "golf_data_expert.csv", "text/csv")
    
if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("📊 GolfShot Analytics 5.0")

tab_saisie, tab_deep, tab_raw = st.tabs(["📝 Saisie Data", "📈 Analyses Avancées", "🗃️ Données Brutes"])

# --- 1. SAISIE ENRICHIE ---
with tab_saisie:
    col1, col2, col3 = st.columns(3)
    with col1:
        club = st.selectbox("Club", CLUBS_ORDER)
    with col2:
        mode = st.radio("Mode", ["Parcours", "Practice"], horizontal=True)
        
    st.markdown("---")
    
    if club == "Putter":
        c1, c2, c3 = st.columns(3)
        with c1: dist = st.number_input("Dist (m)", 0.0, 25.0, 2.0, 0.1)
        with c2: pente = st.selectbox("Pente", ["Plat", "Gauche-Droite", "Droite-Gauche"])
        with c3: res = st.radio("Résultat", ["Dans le trou", "Raté"])
        
        if st.button("💾 Save Putt", type="primary"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': mode, 'club': club,
                'distance': dist, 'pente': pente, 'resultat': res, 
                'type_coup': 'Putt', 'green_touche': False
            })
            st.success("Putt enregistré")
            
    else:
        c1, c2, c3 = st.columns(3)
        with c1: 
            dist = st.number_input("Distance (m)", 0, 300, 135)
            green_touche = st.checkbox("Green Touché (GIR) ?", value=False, help="Cochez si la balle a fini sur le green")
        with c2: 
            direction = st.radio("Axe", ["Gauche", "Centre", "Droite"], horizontal=True)
        with c3: 
            longueur = st.radio("Profondeur", ["Court", "Ok", "Long"], horizontal=True)
            
        if st.button(f"💾 Save {club}", type="primary"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': mode, 'club': club,
                'distance': dist, 'direction': direction, 'longueur': longueur,
                'resultat': f"{direction}/{longueur}", 'type_coup': 'Jeu Long',
                'green_touche': green_touche
            })
            st.success("Coup enregistré")

# --- 2. ANALYSE EXPERT ---
with tab_deep:
    if not st.session_state['coups']:
        st.info("En attente de données...")
    else:
        df = pd.DataFrame(st.session_state['coups'])
        
        # SECTION 1 : PUTTING PRO STATS
        st.markdown("### 🟢 Putting Performance Curve")
        df_putt = df[df['type_coup'] == 'Putt'].copy()
        
        if not df_putt.empty:
            # Création de "Buckets" (Intervalles de distance)
            bins = [0, 1.5, 3, 6, 10, 20]
            labels = ["0-1.5m", "1.5-3m", "3-6m", "6-10m", "+10m"]
            df_putt['zone'] = pd.cut(df_putt['distance'], bins=bins, labels=labels)
            
            # Calcul du % de réussite par zone
            stats_putt = df_putt.groupby('zone', observed=False)['resultat'].apply(lambda x: (x=='Dans le trou').mean() * 100).reset_index()
            stats_putt.columns = ['Zone', 'Reussite_Pct']
            
            col_g, col_d = st.columns([2, 1])
            with col_g:
                st.bar_chart(stats_putt.set_index('Zone'))
            with col_d:
                st.dataframe(stats_putt.style.format({'Reussite_Pct': "{:.1f}%"}))
                st.caption("Comparez ces chiffres aux pros (ex: 0-1.5m > 90% attendu).")
        else:
            st.warning("Pas assez de putts.")

        st.markdown("---")

        # SECTION 2 : GIR & PRÉCISION PAR CLUB
        st.markdown("### 🎯 Analyse Green in Regulation (GIR)")
        df_shots = df[df['type_coup'] == 'Jeu Long'].copy()
        
        if not df_shots.empty:
            # Filtre : On ne garde que les fers et wedges pour le GIR
            clubs_approche = [c for c in CLUBS_ORDER if "Fer" in c or "PW" in c or "°" in c]
            df_app = df_shots[df_shots['club'].isin(clubs_approche)]
            
            if not df_app.empty:
                # KPI : % de GIR par Club
                gir_stats = df_app.groupby('club')['green_touche'].mean() * 100
                
                st.write("Pourcentage de Greens touchés par club :")
                st.bar_chart(gir_stats)
                
                # KPI : Dispersion moyenne (approximation via labels)
                # On considère "Centre/Ok" comme précis.
                df_app['is_precise'] = ((df_app['direction']=='Centre') & (df_app['longueur']=='Ok'))
                precision_stats = df_app.groupby('club')['is_precise'].mean() * 100
                
                with st.expander("Voir les détails chiffrés (GIR & Précision)"):
                    summary = pd.concat([gir_stats, precision_stats], axis=1)
                    summary.columns = ['% GIR', '% Impact Parfait']
                    st.dataframe(summary.style.format("{:.1f}%").background_gradient(cmap="Greens"))
            else:
                st.info("Ajoutez des coups de fers pour voir l'analyse GIR.")

        # SECTION 3 : CONSISTANCE DES DISTANCES (Histogramme)
        st.markdown("---")
        st.markdown("### 📏 Consistance & Étalonnage")
        club_select = st.selectbox("Analyser la régularité du :", df_shots['club'].unique())
        
        subset = df_shots[df_shots['club'] == club_select]
        if len(subset) > 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(subset['distance'], bins=10, color='skyblue', edgecolor='black')
            ax.set_title(f"Distribution des distances : {club_select}")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Fréquence")
            
            # Lignes Moyenne et Médiane
            mean_val = subset['distance'].mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Moy: {mean_val:.1f}m')
            ax.legend()
            
            st.pyplot(fig)
            st.caption("Plus la courbe est étroite et haute, plus vous êtes régulier.")

# --- 3. DONNÉES BRUTES (POUR EXCEL) ---
with tab_raw:
    st.markdown("### 🗃️ Base de données complète")
    st.markdown("Utilisez les filtres ci-dessous pour explorer vos données, puis exportez en CSV via le menu de gauche.")
    
    if st.session_state['coups']:
        df_all = pd.DataFrame(st.session_state['coups'])
        
        # Filtres dynamiques
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            f_club = st.multiselect("Filtrer par Club", df_all['club'].unique())
        with col_f2:
            f_type = st.multiselect("Type de coup", df_all['type_coup'].unique())
            
        # Application des filtres
        df_show = df_all.copy()
        if f_club: df_show = df_show[df_show['club'].isin(f_club)]
        if f_type: df_show = df_show[df_show['type_coup'].isin(f_type)]
        
        st.dataframe(df_show, use_container_width=True)
        st.metric("Nombre de données filtrées", len(df_show))
