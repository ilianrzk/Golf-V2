import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Configuration de la page
st.set_page_config(page_title="GolfShot Pro 4.1", layout="wide")

# --- GESTION DES DONNÉES ---
if 'coups' not in st.session_state:
    st.session_state['coups'] = []

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- LISTE OFFICIELLE DES CLUBS (MODIFIÉE) ---
# Fer 4 remplacé par Fer 3
CLUBS_ORDER = [
    "Driver", "Bois 5", "Hybride", 
    "Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9", 
    "PW", "50°", "55°", "60°", 
    "Putter"
]

# --- SIDEBAR : SAUVEGARDE & GÉNÉRATEUR ---
st.sidebar.title("⚙️ Options")

# 1. Générateur de Données de Test
st.sidebar.markdown("### 🧪 Zone de Test")
if st.sidebar.button("Générer 10 coups/club (Test)"):
    new_data = []
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(5)]
    
    for club in CLUBS_ORDER:
        for _ in range(10): # 10 coups par club
            # Logique réaliste simulée
            if club == 'Putter':
                dist = np.random.uniform(1.0, 15.0)
                res = np.random.choice(["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite"], p=[0.4, 0.15, 0.15, 0.15, 0.15])
                coup = {
                    'date': str(np.random.choice(dates)),
                    'mode': 'Practice',
                    'par_trou': 4,
                    'club': club,
                    'distance': round(dist, 1),
                    'lie': 'Green',
                    'pente': np.random.choice(["Plat", "Gauche-Droite", "Droite-Gauche"]),
                    'denivele': np.random.choice(["Plat", "Montée", "Descente"]),
                    'resultat': res,
                    'type_coup': 'Putt'
                }
            else:
                # Distances approximatives
                if club == 'Driver': base = 220
                elif club == 'Fer 3': base = 180 # Distance estimée Fer 3
                elif club == 'Fer 7': base = 150
                else: base = 80
                
                noise = np.random.randint(-20, 20)
                direction = np.random.choice(["Gauche", "Centre", "Droite"], p=[0.2, 0.6, 0.2])
                longueur = np.random.choice(["Court", "Ok", "Long"], p=[0.2, 0.6, 0.2])
                
                coup = {
                    'date': str(np.random.choice(dates)),
                    'mode': 'Practice',
                    'par_trou': 4,
                    'club': club,
                    'distance': base + noise,
                    'lie': 'Tapis/Herbe',
                    'direction': direction,
                    'longueur': longueur,
                    'resultat': f"{direction}/{longueur}",
                    'type_coup': 'Jeu Long'
                }
            new_data.append(coup)
    
    st.session_state['coups'].extend(new_data)
    st.sidebar.success(f"✅ Données générées (avec Fer 3) !")

# 2. Sauvegarde
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Sauvegarde")
uploaded_file = st.sidebar.file_uploader("Charger CSV", type="csv")
if uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        st.session_state['coups'] = df_loaded.to_dict('records')
        st.sidebar.success("Chargé !")
    except:
        st.sidebar.error("Erreur")

if st.session_state['coups']:
    df_export = pd.DataFrame(st.session_state['coups'])
    csv = convert_df(df_export)
    st.sidebar.download_button("📥 Télécharger CSV", data=csv, file_name='golf_pro_v4_1.csv', mime='text/csv')
    
if st.sidebar.button("🗑️ Reset Total"):
    st.session_state['coups'] = []

# --- INTERFACE PRINCIPALE ---
st.title("🏌️‍♂️ GolfShot Pro 4.1")

# Sélecteur de Mode pour la Saisie
mode_saisie = st.radio("Mode de Saisie :", ["Parcours ⛳", "Practice 🚜"], horizontal=True)
mode_db = "Parcours" if mode_saisie == "Parcours ⛳" else "Practice"

tab1, tab2 = st.tabs(["📝 Saisie des Coups", "🧠 LE COACH (Analyse)"])

# --- ONGLET 1 : SAISIE ---
with tab1:
    st.markdown(f"### Enregistrement - Mode {mode_db}")
    
    col_club, col_dist = st.columns(2)
    with col_club:
        club = st.selectbox("Club", CLUBS_ORDER)
    with col_dist:
        # Contexte différent selon le mode
        if mode_db == "Parcours":
            par_trou = st.selectbox("Par", [3, 4, 5], index=1)
            lie = st.selectbox("Lie", ["Tee", "Fairway", "Rough", "Bunker"])
        else:
            par_trou = 0 
            lie = "Practice"

    # --- LOGIQUE PUTTING ---
    if club == "Putter":
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            dist_putt = st.number_input("Distance (m)", 0.0, 20.0, 1.5, 0.1, format="%.1f")
        with col_p2:
            pente = st.selectbox("Pente", ["Plat", "Gauche-Droite", "Droite-Gauche"])
            deniv = st.selectbox("Dénivelé", ["Plat", "Montée", "Descente"])
        with col_p3:
            res_putt = st.radio("Résultat", ["Dans le trou", "Raté - Court", "Raté - Long", "Raté - Gauche", "Raté - Droite"])
            
        if st.button("Valider Putt", type="primary"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': mode_db,
                'club': club, 'distance': dist_putt, 'lie': 'Green',
                'pente': pente, 'denivele': deniv, 'resultat': res_putt,
                'type_coup': 'Putt', 'par_trou': par_trou
            })
            st.success("Putt enregistré")

    # --- LOGIQUE JEU LONG ---
    else:
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            dist_shot = st.number_input("Distance (m)", 0, 300, 130)
        with col_j2:
            direction = st.radio("Direction", ["Gauche", "Centre", "Droite"], horizontal=True)
            longueur = st.radio("Profondeur", ["Court", "Ok", "Long"], horizontal=True)
            
        if st.button(f"Valider {club}", type="primary"):
            st.session_state['coups'].append({
                'date': str(datetime.date.today()), 'mode': mode_db,
                'club': club, 'distance': dist_shot, 'lie': lie,
                'direction': direction, 'longueur': longueur,
                'resultat': f"{direction}/{longueur}", 'type_coup': 'Jeu Long', 'par_trou': par_trou
            })
            st.success(f"Coup de {club} enregistré")

# --- ONGLET 2 : LE COACH IA ---
with tab2:
    if not st.session_state['coups']:
        st.info("Commencez par saisir des données ou utilisez le bouton 'Générer' dans le menu.")
    else:
        df = pd.DataFrame(st.session_state['coups'])
        
        st.markdown("## 🧠 Analyse Sectorielle")
        
        c_driver, c_fers, c_wedges, c_putt = st.tabs(["🚀 Driving", "⚔️ Fers", "🎯 Wedges", "🟢 Putting"])
        
        # --- COACH DRIVING ---
        with c_driver:
            df_drive = df[df['club'].isin(['Driver', 'Bois 5', 'Hybride'])]
            if not df_drive.empty:
                total = len(df_drive)
                miss_left = len(df_drive[df_drive['direction'] == 'Gauche'])
                miss_right = len(df_drive[df_drive['direction'] == 'Droite'])
                fairway = len(df_drive[df_drive['direction'] == 'Centre'])
                
                col1, col2 = st.columns(2)
                col1.metric("Précision (Fairway)", f"{int(fairway/total*100)}%")
                if total > 0:
                    col1.metric("Distance Moyenne", f"{int(df_drive['distance'].mean())}m")
                
                st.subheader("Diagnostic Driving :")
                if miss_right > miss_left and miss_right > total * 0.4:
                    st.error("⚠️ **Problème : SLICE**")
                    st.write("Conseil : Surveillez votre alignement d'épaules à l'adresse.")
                elif miss_left > miss_right and miss_left > total * 0.4:
                    st.error("⚠️ **Problème : HOOK**")
                    st.write("Conseil : Relâchez la pression du grip.")
                else:
                    st.success("✅ Bon équilibre latéral.")
            else:
                st.warning("Pas de données de bois.")

        # --- COACH FERS (MISE À JOUR FER 3) ---
        with c_fers:
            # Nouvelle liste incluant Fer 3
            fers_list = ["Fer 3", "Fer 5", "Fer 6", "Fer 7", "Fer 8", "Fer 9"]
            df_fers = df[df['club'].isin(fers_list)]
            
            if not df_fers.empty:
                miss_short = len(df_fers[df_fers['longueur'] == 'Court'])
                total = len(df_fers)
                
                st.write(f"Basé sur {total} coups de fers (du 3 au 9).")
                
                fig, ax = plt.subplots(figsize=(5,3))
                def simple_coords(r):
                    return (-1 if r['direction']=='Gauche' else 1 if r['direction']=='Droite' else 0) + np.random.normal(0,0.1), \
                           (-1 if r['longueur']=='Court' else 1 if r['longueur']=='Long' else 0) + np.random.normal(0,0.1)
                coords = df_fers.apply(simple_coords, axis=1, result_type='expand')
                ax.scatter(coords[0], coords[1], alpha=0.5)
                ax.set_title("Dispersion Fers")
                ax.set_xticks([]); ax.set_yticks([])
                col_graph, col_txt = st.columns(2)
                col_graph.pyplot(fig)
                
                with col_txt:
                    if miss_short > total * 0.4:
                        st.warning("📉 **Tendance : Trop Court**")
                        st.write("Conseil : Prenez un club de plus (ex: Fer 6 au lieu de Fer 7) et tapez à 80%.")
                    else:
                        st.success("✅ Bonne gestion des distances.")
            else:
                st.warning("Pas de données de Fers.")

        # --- COACH WEDGES ---
        with c_wedges:
            df_wedges = df[df['club'].isin(['PW', '50°', '55°', '60°'])]
            if not df_wedges.empty:
                st.dataframe(df_wedges.groupby('club')['distance'].agg(['mean', 'std', 'count']).round(1))
            else:
                st.warning("Pas de données de Wedges.")

        # --- COACH PUTTING ---
        with c_putt:
            df_putt = df[df['club'] == 'Putter']
            if not df_putt.empty:
                made = len(df_putt[df_putt['resultat'] == 'Dans le trou'])
                total = len(df_putt)
                st.metric("Taux de réussite global", f"{int(made/total*100)}%")
                
                if 'pente' in df_putt.columns:
                    st.subheader("Analyse Pentes")
                    # On calcule le taux d'ECHEC pour voir la faiblesse
                    res_pente = df_putt.groupby('pente')['resultat'].apply(lambda x: (x != 'Dans le trou').mean() * 100)
                    st.bar_chart(res_pente)
                    st.caption("Barre haute = Pente qui vous pose problème (Taux d'échec).")
            else:
                st.warning("Pas de données de Putting.")
