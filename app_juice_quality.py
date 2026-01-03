import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Prediction Qualite du Jus",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .quality-score {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Charge le modele et les artefacts"""
    try:
        artifacts = joblib.load('best_model_juice.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("Le fichier 'best_model_juice.pkl' n'a pas ete trouve. Veuillez executer le notebook d'entrainement.")
        return None


def predict_quality(model, scaler, encoder, features_df):
    """Fait une prediction de qualite"""
    features_scaled = scaler.transform(features_df)
    prediction_encoded = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    prediction = encoder.inverse_transform(prediction_encoded)
    return prediction[0], prediction_proba[0]


def create_proba_chart(probas, classes):
    """Cree un graphique des probabilites par classe"""
    df_proba = pd.DataFrame({
        'Qualite': [str(c) for c in classes],
        'Probabilite': probas * 100
    })
    
    fig = px.bar(
        df_proba,
        x='Qualite',
        y='Probabilite',
        color='Probabilite',
        color_continuous_scale='Blues',
        title='Probabilites par classe de qualite'
    )
    fig.update_layout(
        xaxis_title='Classe de Qualite',
        yaxis_title='Probabilite (%)',
        showlegend=False
    )
    return fig


def create_radar_chart(features_dict, feature_names):
    """Cree un graphique radar des caracteristiques"""
    values = list(features_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=feature_names,
        fill='toself',
        name='Echantillon',
        line_color='#1E88E5'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=False,
        title='Profil de l\'echantillon'
    )
    return fig


def main():
    artifacts = load_model()
    
    if artifacts is None:
        st.stop()
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoder = artifacts['encoder']
    feature_names = artifacts['feature_names']
    classes = artifacts['classes']
    model_name = artifacts['model_name']
    
    st.markdown('<p class="main-header">Prediction de la Qualite du Jus</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Modele utilise : {model_name}</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("A propos")
        st.write("""
        Cette application utilise un modele de Machine Learning 
        pour predire la qualite du jus en fonction de ses 
        caracteristiques physico chimiques.
        """)
        
        st.header("Features utilisees")
        for i, feat in enumerate(feature_names, 1):
            st.write(f"{i}. {feat}")
        
        st.header("Classes de qualite")
        st.write(f"De {min(classes)} (faible) a {max(classes)} (excellente)")
    
    tab1, tab2 = st.tabs(["Prediction Manuelle", "Prediction par Fichier"])
    
    with tab1:
        st.header("Entrez les caracteristiques du jus")
        
        col1, col2 = st.columns(2)
        
        features_input = {}
        
        with col1:
            features_input['fixed acidity'] = st.number_input(
                "Fixed Acidity (g/L)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.1,
                help="Acidite fixe du jus"
            )
            
            features_input['volatile acidity'] = st.number_input(
                "Volatile Acidity (g/L)",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.01,
                help="Acidite volatile"
            )
            
            features_input['residual sugar'] = st.number_input(
                "Residual Sugar (g/L)",
                min_value=0.0,
                max_value=70.0,
                value=5.0,
                step=0.5,
                help="Sucre residuel"
            )
            
            features_input['chlorides'] = st.number_input(
                "Chlorides (g/L)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Concentration en chlorures"
            )
        
        with col2:
            features_input['free sulfur dioxide'] = st.number_input(
                "Free Sulfur Dioxide (mg/L)",
                min_value=0.0,
                max_value=300.0,
                value=30.0,
                step=1.0,
                help="Dioxyde de soufre libre"
            )
            
            features_input['total sulfur dioxide'] = st.number_input(
                "Total Sulfur Dioxide (mg/L)",
                min_value=0.0,
                max_value=500.0,
                value=100.0,
                step=1.0,
                help="Dioxyde de soufre total"
            )
            
            features_input['pH'] = st.number_input(
                "pH",
                min_value=2.0,
                max_value=5.0,
                value=3.2,
                step=0.01,
                help="Niveau de pH"
            )
            
            features_input['alcohol'] = st.number_input(
                "Alcohol (%)",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                help="Teneur en alcool"
            )
        
        st.markdown("---")
        
        if st.button("Predire la Qualite", type="primary"):
            # Preparation des donnees
            features_df = pd.DataFrame([features_input])
            features_df = features_df[feature_names]
            
            prediction, probas = predict_quality(model, scaler, encoder, features_df)
            
            # Affichage des resultats
            col_result1, col_result2 = st.columns([1, 2])
            
            with col_result1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Qualite Predite")
                st.markdown(f'<p class="quality-score">{prediction}</p>', unsafe_allow_html=True)
                
                max_proba = max(probas) * 100
                st.metric("Confiance", f"{max_proba:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_result2:
                fig_proba = create_proba_chart(probas, classes)
                st.plotly_chart(fig_proba, use_container_width=True)
            
            # Graphique radar
            st.subheader("Profil de l'echantillon")
            fig_radar = create_radar_chart(features_input, list(features_input.keys()))
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        st.header("Prediction par lot")
        st.write("Chargez un fichier CSV avec les caracteristiques pour faire des predictions en lot.")
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV",
            type=['csv'],
            help="Le fichier doit contenir les colonnes: " + ", ".join(feature_names)
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                # Verification des colonnes
                missing_cols = [col for col in feature_names if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"Colonnes manquantes : {missing_cols}")
                else:
                    st.success(f"Fichier charge avec succes ! {len(df_upload)} echantillons detectes.")
                    
                    st.subheader("Apercu des donnees")
                    st.dataframe(df_upload.head())
                    
                    if st.button("Lancer les predictions", type="primary"):
                        features_df = df_upload[feature_names]
                        features_scaled = scaler.transform(features_df)
                        predictions_encoded = model.predict(features_scaled)
                        predictions = encoder.inverse_transform(predictions_encoded)
                        
                        probas = model.predict_proba(features_scaled)
                        max_probas = np.max(probas, axis=1) * 100
                        
                        df_results = df_upload.copy()
                        df_results['Qualite_Predite'] = predictions
                        df_results['Confiance_%'] = max_probas.round(1)
                        
                        st.subheader("Resultats des predictions")
                        st.dataframe(df_results)
                        
                        fig_dist = px.histogram(
                            df_results,
                            x='Qualite_Predite',
                            title='Distribution des qualites predites',
                            color='Qualite_Predite'
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Telechargement des resultats
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="Telecharger les resultats (CSV)",
                            data=csv,
                            file_name="predictions_qualite.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {str(e)}")
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
            Application developpee pour la prediction de qualite du jus | Modele XGBoost
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
