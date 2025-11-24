import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot 7.0 Coach IA", layout="wide")

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

if st.sidebar.button("Générer Données (Test Contact)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(10)]
    
    for _ in range(100):
        club = np.random.choice(["Driver", "Fer 7", "PW"])
        mode = np.random.choice(["Practice", "Parcours"])
        
        # Simulation réaliste
        if mode == "Practice":
            dist_base = 150 if club == "Fer 7" else (230 if club == "Driver" else 100)
            dist = np.random.normal(dist_base, 5)
            lat = np.random.normal(0, 1) # Score lateral faible (0-2)
            contact = np.random.choice(["Bon Contact", "Gratte", "Top"], p=[0.7, 0.2, 0.1])
        else:
            dist_base = 145 if club == "Fer 7" else (220 if club == "Driver" else 95)
            dist = np.random.normal(dist_base, 12) # Plus de variance distance
            lat = np.random.normal(0, 2) # Score lateral plus élevé
            contact = np.random.choice(["Bon Contact", "Gratte", "Top"], p=[0.5, 0.25, 0.25])

        # Conversion score latéral float vers int 0-5 pour l'affichage
        score_lat_int = min(5, int(abs(lat)))
        direction = "Centre" if score_lat_int == 0 else ("Gauche" if lat < 0 else "Droite")
        
        new_data.append({
            'date': str(np.random.choice(dates)), 'mode': mode, 'club': club,
            'distance': round(dist, 1), 'direction': direction,
            'score_lateral': score_lat_int,
            'contact': contact, # Nouvelle donnée
            'type_coup': 'Jeu Long', 'longueur': 'Ok'
        })
    
    st.session_state['coups'].extend(new_data)
    st.sidebar.success("Données de test générées !")

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
    st.sidebar.download_button("📥 Sauvegarder CSV", convert_df(df_ex), "golf_v7.csv", "text/csv")
    if st.sidebar.button("🗑️ Reset"): st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot 7.0 : Coach IA & Ball Striking")

tab_saisie, tab_coach, tab_visu = st.tabs(["📝 Saisie", "🤖 Coach IA Technique", "🎯 Dispersion & Cercles"])

# --- ONGLET 1 : SAISIE (AVEC CONTACT) ---
with tab_saisie:
    col_mode, col_club = st.columns(2)
    with col_mode:
        mode_active = st.radio("Mode", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
        mode_db = "Parcours" if "Parcours" in mode_active else "Practice"
    with col_club:
        club = st.selectbox("Club", CLUBS_ORDER)

    st.markdown("---")
    
    if club == "Putter":
        st.info("Saisie Putt simplifiée pour cet exemple.")
        if st.button("Enregistrer Putt"):
             st.session_state['coups'].append({'date': str(datetime.date.today()), 'mode': mode_db, 'club': 'Putter', 'type_coup': 'Putt', 'contact': 'Bon Contact'})
             st.success("Putt noté")
    else:
        # SAISIE JEU LONG
        c1, c2, c3 = st.columns(3)
        with c1:
            dist = st.number_input("Distance (m)", 0, 320, 140)
        with c2:
            # NOUVEAUTÉ : QUALITÉ DE CONTACT
            contact = st.selectbox("Sensation / Contact", ["Bon Contact", "Gratte (Sol avant)", "Top (Balle coiffée)"])
        with c3:
            direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
        
        if direction != "Centre":
            score_lat = st.slider("Écart Latéral (0=Axe, 5=Perdue)", 1, 5, 2)
        else:
            score_lat = 0

        if st.button(f"Valider {club}", type="primary"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()),
                'mode': mode_db,
                'club': club,
                'distance': dist,
                'direction': direction,
                'score_lateral': score_lat,
                'contact': contact, # Sauvegarde du contact
                'type_coup': 'Jeu Long',
                'longueur': 'Ok' 
            })
            st.success(f"Coup enregistré ! ({contact})")

# --- ONGLET 2 : LE COACH IA TECHNIQUE ---
with tab_coach:
    if not st.session_state['coups']:
        st.info("En attente de données...")
    else:
        df = pd.DataFrame(st.session_state['coups'])
        df = df[df['type_coup'] == 'Jeu Long']
        
        if not df.empty:
            st.header("🤖 L'analyse du Coach")
            
            # 1. ANALYSE DES CONTACTS
            st.subheader("1. Qualité de Frappe")
            
            # Calculs
            total = len(df)
            nb_gratte = len(df[df['contact'].str.contains("Gratte")])
            nb_top = len(df[df['contact'].str.contains("Top")])
            nb_bon = len(df[df['contact'].str.contains("Bon")])
            
            col_res, col_conseil = st.columns([1, 2])
            
            with col_res:
                # Pie Chart simple
                labels = ['Bon', 'Gratte', 'Top']
                sizes = [nb_bon, nb_gratte, nb_top]
                fig1, ax1 = plt.subplots(figsize=(3,3))
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999','#ffcc99'])
                st.pyplot(fig1)
            
            with col_conseil:
                st.markdown("### 💡 Conseil Technique Personnalisé")
                
                pct_gratte = nb_gratte / total
                pct_top = nb_top / total
                
                if pct_gratte > 0.25: # Si plus de 25% de grattes
                    st.error(f"⚠️ **Alerte Gratte ({int(pct_gratte*100)}%)** : Vous touchez le sol avant la balle.")
                    st.info("""
                    **Pourquoi ?** Souvent, votre poids reste sur le pied arrière (droit pour un droitier) à la descente, ou vous 'jetez' les mains trop tôt.
                    \n**Exercice :** Tapez des balles avec les pieds joints, ou essayez de finir votre swing en équilibre complet sur le pied avant.
                    """)
                
                elif pct_top > 0.25: # Si plus de 25% de tops
                    st.warning(f"⚠️ **Alerte Top ({int(pct_top*100)}%)** : Vous tapez le haut de la balle.")
                    st.info("""
                    **Pourquoi ?** Vous vous redressez probablement pendant l'impact ou vos bras se plient (ailes de poulet).
                    \n**Exercice :** Gardez la posture ! Imaginez que votre poitrine doit continuer à regarder le sol une seconde après avoir frappé la balle.
                    """)
                
                else:
                    st.success("✅ **Excellent Contact** : Vous centrez bien la balle majoritairement. La priorité est maintenant la direction (face de club).")

            # 2. CONSEIL DISPERSION
            st.divider()
            st.subheader("2. Consistance")
            avg_disp = df['score_lateral'].mean()
            if avg_disp > 2.5:
                st.write("Votre dispersion latérale est élevée (Score > 2.5).")
                st.write("🧠 **Conseil Mental :** Arrêtez de viser les drapeaux. Visez le centre géométrique du green. Cela vous donne une marge d'erreur à gauche et à droite.")

# --- ONGLET 3 : DISPERSION AVEC CERCLES ---
with tab_visu:
    st.header("🎯 Visualisation Avancée")
    
    if st.session_state['coups']:
        df = pd.DataFrame(st.session_state['coups'])
        df_graph = df[df['type_coup'] == 'Jeu Long']
        
        filter_club = st.selectbox("Analyser Club", df_graph['club'].unique(), key="viz_club")
        data_plot = df_graph[df_graph['club'] == filter_club]
        
        if not data_plot.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Préparation des coordonnées X (Latéral)
            def get_x_coord(row):
                val = row['score_lateral']
                # On "spatialise" le score 0-5
                factor = 5 # Echelle arbitraire pour le graph (1 score point = 5 mètres approx)
                jitter = np.random.normal(0, 1.5)
                if row['direction'] == 'Gauche': return (val * -factor) + jitter
                if row['direction'] == 'Droite': return (val * factor) + jitter
                return 0 + jitter

            data_plot['x_viz'] = data_plot.apply(get_x_coord, axis=1)
            
            # Séparation
            prac = data_plot[data_plot['mode'] == 'Practice']
            parc = data_plot[data_plot['mode'] == 'Parcours']
            
            # Fonction pour dessiner l'ellipse de dispersion
            def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
                if len(x) < 2: return # Pas assez de points
                cov = np.cov(x, y)
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                              width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                              angle=np.rad2deg(np.arccos(v[0, 0])), **kwargs)
                ell.set_facecolor(facecolor)
                ax.add_artist(ell)

            # Plot Points et Cercles
            if not prac.empty:
                ax.scatter(prac['x_viz'], prac['distance'], c='blue', alpha=0.6, label='Practice', edgecolors='white')
                # Cercle Bleu (Practice)
                draw_confidence_ellipse(prac['x_viz'], prac['distance'], ax, n_std=2, facecolor='blue', alpha=0.1, edgecolor='blue', linestyle='--')
            
            if not parc.empty:
                ax.scatter(parc['x_viz'], parc['distance'], c='red', marker='X', s=80, alpha=0.7, label='Parcours')
                # Cercle Rouge (Parcours)
                draw_confidence_ellipse(parc['x_viz'], parc['distance'], ax, n_std=2, facecolor='red', alpha=0.1, edgecolor='red', linestyle='--')

            ax.set_title(f"Comparaison des Zones de Dispersion : {filter_club}")
            ax.set_xlabel(" Dispersion Latérale (Gauche <---> Droite)")
            ax.set_ylabel("Distance (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ajuster les limites pour que tout soit visible
            ax.autoscale()
            
            st.pyplot(fig)
            st.info("🔵 **Zone Bleue (Practice)** vs 🔴 **Zone Rouge (Parcours)**.\nLa zone colorée représente l'endroit où atterrissent 95% de vos balles (2 écarts-types).")
            
            # Interprétation IA du graphique
            if not parc.empty and not prac.empty:
                std_prac = prac['distance'].std()
                std_parc = parc['distance'].std()
                if std_parc > std_prac * 1.5:
                    st.write(f"⚠️ Votre zone de dispersion est **{int((std_parc/std_prac)*100)-100}% plus large** sur le parcours.")
