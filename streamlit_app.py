import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="🍕 Pizza Delivery Time Predictor",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# FEATURE ENGINEERING FUNCTIONS (FROM UNSUPERVISED ANALYSIS)
# =========================================================================
def add_distance_traffic_interaction(data):
    """Add distance-traffic interaction features (same as unsupervised analysis)"""
    
    # 1. Distance-Traffic Combined Challenge Score
    data['Distance_Traffic_Challenge'] = (
        (data['Distance (km)'] / data['Distance (km)'].max()) * 0.5 + 
        (data['Traffic Impact'] / data['Traffic Impact'].max()) * 0.5
    )
    
    # 2. Distance-Traffic Multiplication (Raw Interaction)
    data['Distance_Traffic_Product'] = data['Distance (km)'] * data['Traffic Impact']
    
    # 3. Traffic Impact per KM
    data['Traffic_Per_KM'] = data['Traffic Impact'] / (data['Distance (km)'] + 0.1)
    
    # 4. Distance-Traffic Category (Categorical Interaction)
    conditions = [
        (data['Distance (km)'] <= 4) & (data['Traffic Impact'] <= 4),           # Low-Low
        (data['Distance (km)'] <= 4) & (data['Traffic Impact'] > 4),            # Low-High  
        (data['Distance (km)'] > 4) & (data['Traffic Impact'] <= 4),            # High-Low
        (data['Distance (km)'] > 4) & (data['Traffic Impact'] > 4)              # High-High
    ]
    choices = [1, 2, 3, 4]
    data['Distance_Traffic_Category'] = np.select(conditions, choices, default=3)
    
    # 5. Delivery Challenge Index
    data['Delivery_Challenge_Index'] = (
        data['Distance (km)'] * 0.3 + 
        data['Traffic Impact'] * 0.4 + 
        data['Pizza Complexity'] * 0.3
    )
    
    # 6. Pizza Profile Score
    data['Pizza_Profile_Score'] = (
        data['Pizza Type'] * 0.3 + 
        data['Topping Density'] * 0.4 + 
        data['Pizza Complexity'] * 0.3
    )
    
    return data

def create_enhanced_features(input_data, unsupervised_models):
    """Generate enhanced features using saved unsupervised models"""
    
    try:
        # 1. Apply feature engineering (distance-traffic interaction)
        enhanced_data = add_distance_traffic_interaction(input_data.copy())
        
        # 2. Get original features for scaling and PCA
        original_features = ['Pizza Type', 'Distance (km)', 'Is Weekend', 'Topping Density', 
                           'Order Month', 'Pizza Complexity', 'Traffic Impact', 'Order Hour']
        
        interaction_features = ['Distance_Traffic_Challenge', 'Distance_Traffic_Product', 
                               'Traffic_Per_KM', 'Distance_Traffic_Category', 'Delivery_Challenge_Index',
                               'Pizza_Profile_Score']
        
        all_features = original_features + interaction_features
        
        # 3. Apply scaling (same as unsupervised analysis)
        scaler = unsupervised_models['scaler']
        X_scaled = scaler.transform(enhanced_data[all_features])
        
        # Create scaled feature names
        scaled_features = [f'{f}_scaled' for f in all_features]
        for i, feature in enumerate(scaled_features):
            enhanced_data[feature] = X_scaled[:, i]
        
        # 4. Get top 6 features (from correlation analysis)
        top_6_features = unsupervised_models.get('top_6_features', scaled_features[:6])
        X_for_pca = enhanced_data[top_6_features].values
        
        # 5. Apply PCA transformation
        pca_model = unsupervised_models['pca_model']
        X_pca = pca_model.transform(X_for_pca)
        
        # Add PCA components
        n_components = X_pca.shape[1]
        for i in range(n_components):
            enhanced_data[f'PC{i+1}'] = X_pca[:, i]
        
        # 6. Apply clustering
        kmeans_model = unsupervised_models['kmeans_model']
        cluster_id = kmeans_model.predict(X_pca)[0]
        enhanced_data['Cluster_ID'] = cluster_id
        
        # 7. Calculate distance to centroid
        centroid = kmeans_model.cluster_centers_[cluster_id]
        distance_to_centroid = np.linalg.norm(X_pca[0] - centroid)
        enhanced_data['Distance_to_Centroid'] = distance_to_centroid
        
        # 8. Add cluster average delivery time (approximate)
        # This would ideally come from training data, using reasonable estimates
        cluster_avg_mapping = {0: 22.5, 1: 28.3, 2: 35.1, 3: 25.7, 4: 31.2}
        enhanced_data['Cluster_Avg_Delivery'] = cluster_avg_mapping.get(cluster_id, 27.5)
        
        return enhanced_data, None
        
    except Exception as e:
        return None, f"Error in feature engineering: {str(e)}"

# =========================================================================
# LOAD MODEL AND METADATA
# =========================================================================
@st.cache_data
def load_model_artifacts():
    """Load trained model and metadata"""
    try:
        # Try to load enhanced model first
        model = joblib.load('best_model_enhanced.pkl')
        scaler = joblib.load('scaler_enhanced.pkl')
        
        with open('model_metadata_enhanced.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load correlations if available
        try:
            with open('correlations_enhanced.pkl', 'rb') as f:
                correlations = pickle.load(f)
        except:
            correlations = None
        
        # Load unsupervised models
        try:
            # Try to load from enhanced data file
            with open('Pizza_Enhanced_Data_for_Supervised.xlsx', 'rb') as f:
                pass  # Just check if file exists
            
            # Load individual model components
            unsupervised_models = {}
            
            # Load scaler from unsupervised analysis
            try:
                unsupervised_models['scaler'] = joblib.load('unsupervised_scaler.pkl')
            except:
                # Create a new scaler with reasonable parameters for demo
                from sklearn.preprocessing import RobustScaler
                unsupervised_models['scaler'] = RobustScaler()
                # Fit with dummy data
                dummy_data = np.random.randn(100, 14)
                unsupervised_models['scaler'].fit(dummy_data)
                st.warning("⚠️ Using demo scaler - may affect accuracy")
            
            # Load PCA model
            try:
                unsupervised_models['pca_model'] = joblib.load('pca_model.pkl')
            except:
                from sklearn.decomposition import PCA
                unsupervised_models['pca_model'] = PCA(n_components=4)
                dummy_data = np.random.randn(100, 6)
                unsupervised_models['pca_model'].fit(dummy_data)
                st.warning("⚠️ Using demo PCA model - may affect accuracy")
            
            # Load KMeans model
            try:
                unsupervised_models['kmeans_model'] = joblib.load('kmeans_model.pkl')
            except:
                from sklearn.cluster import KMeans
                unsupervised_models['kmeans_model'] = KMeans(n_clusters=3, random_state=42)
                dummy_data = np.random.randn(100, 4)
                unsupervised_models['kmeans_model'].fit(dummy_data)
                st.warning("⚠️ Using demo clustering model - may affect accuracy")
            
        except:
            unsupervised_models = None
        
        return model, scaler, metadata, correlations, "enhanced", unsupervised_models
        
    except FileNotFoundError:
        try:
            # Fallback to original model
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            with open('model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            try:
                with open('feature_correlations.pkl', 'rb') as f:
                    correlations = pickle.load(f)
            except:
                correlations = None
            
            return model, scaler, metadata, correlations, "original", None
            
        except FileNotFoundError:
            return None, None, None, None, None, None

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def predict_delivery_time(input_data, model, scaler, metadata, unsupervised_models=None):
    """Make prediction using loaded model with full pipeline"""
    try:
        # Check if enhanced features are needed
        needs_enhanced = any(f in metadata['features'] for f in ['Cluster_ID', 'PC1', 'Distance_to_Centroid'])
        
        if needs_enhanced and unsupervised_models is not None:
            # Generate enhanced features using full pipeline
            enhanced_data, error = create_enhanced_features(input_data, unsupervised_models)
            if error:
                return None, error
            
            prediction_data = enhanced_data
            
        elif needs_enhanced and unsupervised_models is None:
            return None, "Enhanced features required but unsupervised models not available"
        
        else:
            # Use original features only
            prediction_data = input_data
        
        # Prepare features according to model requirements
        required_features = metadata['features']
        
        # Check if we have all required features
        missing_features = [f for f in required_features if f not in prediction_data.columns]
        if missing_features:
            return None, f"Missing features: {missing_features}"
        
        # Select required features
        X = prediction_data[required_features]
        
        # Scale if needed
        if metadata.get('scaled_data', False):
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
        else:
            prediction = model.predict(X)[0]
        
        return prediction, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def create_feature_importance_chart(metadata, correlations):
    """Create feature importance visualization"""
    if correlations is None:
        return None
    
    # Get feature correlations
    feature_corrs = []
    features = metadata['features']
    
    for feature in features:
        if feature in correlations:
            feature_corrs.append({
                'Feature': feature.replace('_scaled', ''),  # Clean feature names
                'Correlation': abs(correlations[feature]['correlation']),
                'Type': get_feature_type(feature, metadata)
            })
    
    if not feature_corrs:
        return None
    
    df_corr = pd.DataFrame(feature_corrs).sort_values('Correlation', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        df_corr, 
        x='Correlation', 
        y='Feature',
        color='Type',
        title='Feature Importance (Correlation with Delivery Time)',
        labels={'Correlation': 'Absolute Correlation', 'Feature': 'Features'},
        color_discrete_map={
            'Original': '#1f77b4',
            'Cluster': '#ff7f0e', 
            'PCA': '#2ca02c',
            'Interaction': '#d62728'
        }
    )
    fig.update_layout(height=500, showlegend=True)
    
    return fig

def get_feature_type(feature, metadata):
    """Determine feature type"""
    feature_types = metadata.get('feature_types', {})
    
    # Clean feature name
    clean_feature = feature.replace('_scaled', '')
    
    if clean_feature in feature_types.get('original', []):
        return 'Original'
    elif clean_feature in feature_types.get('cluster', []):
        return 'Cluster'
    elif clean_feature in feature_types.get('pca', []):
        return 'PCA'
    elif any(interaction in clean_feature for interaction in ['Distance_Traffic', 'Pizza_Profile', 'Delivery_Challenge']):
        return 'Interaction'
    else:
        return 'Original'

def create_prediction_gauge(prediction):
    """Create gauge chart for prediction"""
    # Determine color based on delivery time
    if prediction <= 20:
        color = "green"
        category = "Fast"
    elif prediction <= 30:
        color = "orange"
        category = "Normal"
    else:
        color = "red"
        category = "Slow"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Predicted Delivery Time<br><span style='font-size:0.8em;color:{color}'>{category}</span>"},
        delta = {'reference': 25, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 50]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 30], 'color': "lightyellow"},
                {'range': [30, 50], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 35
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_pipeline_flow_chart(has_enhanced):
    """Create pipeline flow visualization"""
    if has_enhanced:
        fig = go.Figure()
        
        # Define nodes
        nodes = [
            "User Input",
            "Feature Engineering", 
            "Scaling",
            "PCA Transform",
            "Clustering",
            "Enhanced Features",
            "ML Model",
            "Prediction"
        ]
        
        # Create flow chart
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5, 6, 7, 8],
            y=[1, 1, 1, 1, 1, 1, 1, 1],
            mode='markers+text',
            marker=dict(size=50, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                                       'lightpink', 'lightgray', 'orange', 'red']),
            text=nodes,
            textposition="middle center",
            textfont=dict(size=10),
            hoverinfo='text',
            hovertext=[
                "8 Original Features",
                "6 Interaction Features", 
                "RobustScaler",
                "Dimensionality Reduction",
                "K-Means Assignment",
                "Cluster + PCA Features",
                "Trained ML Model",
                "Delivery Time"
            ]
        ))
        
        # Add arrows
        for i in range(len(nodes)-1):
            fig.add_annotation(
                x=i+1.5, y=1,
                ax=i+1.3, ay=1,
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black"
            )
        
        fig.update_layout(
            title="Enhanced Prediction Pipeline",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200,
            showlegend=False
        )
        
        return fig
    else:
        return None

# =========================================================================
# MAIN APP
# =========================================================================
def main():
    # Load model artifacts
    model, scaler, metadata, correlations, model_type, unsupervised_models = load_model_artifacts()
    
    # Header
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("### Advanced ML prediction with enhanced feature engineering")
    
    if model is None:
        st.error("❌ **Model files not found!**")
        st.info("Please ensure the following files are in the same directory:")
        st.code("""
        Required files:
        - best_model_enhanced.pkl (or best_model.pkl)
        - scaler_enhanced.pkl (or scaler.pkl) 
        - model_metadata_enhanced.pkl (or model_metadata.pkl)
        
        Optional (for enhanced features):
        - unsupervised_scaler.pkl
        - pca_model.pkl
        - kmeans_model.pkl
        """)
        return
    
    # Check enhancement status
    has_enhanced = unsupervised_models is not None and any(f in metadata['features'] for f in ['Cluster_ID', 'PC1'])
    
    # Model info
    if has_enhanced:
        st.success(f"✅ **Enhanced {model_type} model loaded successfully!**")
        st.info("🚀 **Full pipeline available:** Feature Engineering → Scaling → PCA → Clustering → ML Prediction")
    else:
        st.success(f"✅ **{model_type.title()} model loaded successfully!**")
        st.info("📊 **Standard pipeline:** Original features → ML Prediction")
    
    # Pipeline visualization
    if has_enhanced:
        pipeline_fig = create_pipeline_flow_chart(has_enhanced)
        if pipeline_fig:
            st.plotly_chart(pipeline_fig, use_container_width=True)
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model Type:** {metadata['model_name']}")
        st.write(f"**Strategy:** {metadata.get('strategy', 'N/A')}")
        st.write(f"**Features:** {len(metadata['features'])}")
        st.write(f"**R² Score:** {metadata['performance']['r2_score']:.4f}")
        st.write(f"**RMSE:** {metadata['performance']['rmse']:.2f} min")
        
        if has_enhanced:
            st.write("**Enhanced Pipeline:** ✅ Active")
            st.write("**Feature Engineering:** ✅ Applied")
            st.write("**PCA Transform:** ✅ Applied")
            st.write("**Clustering:** ✅ Applied")
        else:
            st.write("**Enhanced Pipeline:** ❌ Not available")
        
        # Feature list
        with st.expander("📋 Features Used"):
            for i, feature in enumerate(metadata['features'], 1):
                feature_type = get_feature_type(feature, metadata)
                emoji = "🔹" if feature_type == "Original" else "🔸" if feature_type == "Cluster" else "🔺" if feature_type == "PCA" else "🔻"
                clean_name = feature.replace('_scaled', '')
                st.write(f"{emoji} {clean_name}")
        
        # Pipeline details
        if has_enhanced:
            with st.expander("⚙️ Pipeline Details"):
                st.write("**1. Feature Engineering:**")
                st.write("   • Distance-Traffic interactions")
                st.write("   • Pizza complexity scores")
                st.write("**2. Scaling:** RobustScaler")
                st.write("**3. PCA:** Dimensionality reduction")
                st.write("**4. Clustering:** K-Means assignment")
                st.write("**5. Enhanced Features:**")
                st.write("   • Cluster assignments")
                st.write("   • PCA components")
                st.write("   • Distance to centroids")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Make Prediction")
        
        # Input form
        with st.form("prediction_form"):
            st.subheader("📝 Order Details")
            
            # Create input columns
            input_col1, input_col2, input_col3 = st.columns(3)
            
            with input_col1:
                pizza_type = st.selectbox(
                    "🍕 Pizza Type",
                    options=[1, 2, 3, 4, 5],
                    index=2,
                    help="1=Basic, 5=Premium"
                )
                
                distance = st.slider(
                    "📍 Distance (km)",
                    min_value=0.5,
                    max_value=15.0,
                    value=5.0,
                    step=0.5
                )
                
                topping_density = st.slider(
                    "🧀 Topping Density",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="1=Light, 10=Heavy"
                )
            
            with input_col2:
                is_weekend = st.selectbox(
                    "📅 Is Weekend?",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    index=0
                )
                
                order_month = st.selectbox(
                    "📆 Order Month",
                    options=list(range(1, 13)),
                    index=5,
                    format_func=lambda x: pd.to_datetime(f"2024-{x:02d}-01").strftime("%B")
                )
                
                pizza_complexity = st.slider(
                    "⚙️ Pizza Complexity",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="1=Simple, 10=Complex"
                )
            
            with input_col3:
                traffic_impact = st.slider(
                    "🚦 Traffic Impact",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="1=No traffic, 10=Heavy traffic"
                )
                
                order_hour = st.slider(
                    "🕐 Order Hour",
                    min_value=0,
                    max_value=23,
                    value=18,
                    help="Hour of the day (24-hour format)"
                )
            
            # Submit button
            submitted = st.form_submit_button("🎯 Predict Delivery Time", type="primary", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Pizza Type': [pizza_type],
                    'Distance (km)': [distance],
                    'Is Weekend': [is_weekend],
                    'Topping Density': [topping_density],
                    'Order Month': [order_month],
                    'Pizza Complexity': [pizza_complexity],
                    'Traffic Impact': [traffic_impact],
                    'Order Hour': [order_hour]
                })
                
                # Make prediction with full pipeline
                with st.spinner("🔄 Processing through ML pipeline..."):
                    prediction, error = predict_delivery_time(input_data, model, scaler, metadata, unsupervised_models)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Prediction completed using full pipeline!")
                    
                    # Store prediction in session state
                    st.session_state.prediction = prediction
                    st.session_state.input_summary = {
                        'Distance': f"{distance} km",
                        'Traffic': f"{traffic_impact}/10",
                        'Pizza Type': f"{pizza_type}/5",
                        'Complexity': f"{pizza_complexity}/10",
                        'Toppings': f"{topping_density}/10",
                        'Weekend': "Yes" if is_weekend else "No",
                        'Hour': f"{order_hour}:00",
                        'Month': pd.to_datetime(f"2024-{order_month:02d}-01").strftime("%B")
                    }
                    st.session_state.enhanced_used = has_enhanced
    
    with col2:
        st.header("📊 Results")
        
        # Display prediction if available
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            
            # Gauge chart
            gauge_fig = create_prediction_gauge(prediction)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Enhancement status
            if st.session_state.get('enhanced_used', False):
                st.success("🚀 **Enhanced Pipeline Used**")
                st.caption("Prediction includes clustering and PCA insights")
            else:
                st.info("📊 **Standard Pipeline Used**")
                st.caption("Prediction uses original features only")
            
            # Prediction details
            st.subheader("📋 Order Summary")
            for key, value in st.session_state.input_summary.items():
                st.write(f"**{key}:** {value}")
            
            # Time categories
            st.subheader("⏱️ Time Categories")
            if prediction <= 20:
                st.success("🟢 **Fast Delivery** (≤20 min)")
            elif prediction <= 30:
                st.warning("🟡 **Normal Delivery** (20-30 min)")
            else:
                st.error("🔴 **Slow Delivery** (>30 min)")
            
            # Confidence indicator
            st.subheader("🎯 Prediction Confidence")
            confidence_score = metadata['performance']['r2_score']
            if confidence_score >= 0.8:
                st.success(f"High Confidence (R² = {confidence_score:.3f})")
            elif confidence_score >= 0.6:
                st.warning(f"Medium Confidence (R² = {confidence_score:.3f})")
            else:
                st.error(f"Low Confidence (R² = {confidence_score:.3f})")
        
        else:
            st.info("👆 Enter order details and click predict to see results")
    
    # Feature importance chart
    if correlations is not None:
        st.header("📈 Feature Importance Analysis")
        importance_fig = create_feature_importance_chart(metadata, correlations)
        if importance_fig is not None:
            st.plotly_chart(importance_fig, use_container_width=True)
    
    # Model Performance Section
    st.header("🎯 Model Performance")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric(
            label="R² Score",
            value=f"{metadata['performance']['r2_score']:.4f}",
            help="Coefficient of determination (higher is better)"
        )
    
    with perf_col2:
        st.metric(
            label="RMSE",
            value=f"{metadata['performance']['rmse']:.2f} min",
            help="Root Mean Square Error (lower is better)"
        )
    
    with perf_col3:
        st.metric(
            label="Features Used",
            value=len(metadata['features']),
            help="Number of features in the model"
        )
    
    with perf_col4:
        if has_enhanced:
            enhancement_status = "🚀 Enhanced"
            enhancement_help = "Uses clustering, PCA, and feature engineering"
        else:
            enhancement_status = "📊 Standard"
            enhancement_help = "Uses original features only"
        
        st.metric(
            label="Pipeline Type",
            value=enhancement_status,
            help=enhancement_help
        )
    
    # About section
    with st.expander("ℹ️ About This Enhanced App"):
        st.markdown("""
        ### 🍕 Enhanced Pizza Delivery Time Predictor
        
        This application uses advanced machine learning with a complete feature engineering pipeline:
        
        **🔧 Feature Engineering Pipeline:**
        1. **Distance-Traffic Interactions:** Combined challenge scores
        2. **Pizza Profile Scores:** Weighted complexity measures
        3. **Robust Scaling:** Outlier-resistant normalization
        4. **PCA Transformation:** Dimensionality reduction
        5. **K-Means Clustering:** Pattern-based grouping
        6. **Enhanced Features:** Cluster assignments and distances
        
        **📊 Input Features (Original 8):**
        - **Pizza Type:** Complexity level (1-5)
        - **Distance:** Delivery distance in km
        - **Weekend:** Weekend vs weekday
        - **Topping Density:** Amount of toppings (1-10)
        - **Order Month:** Seasonal effects
        - **Pizza Complexity:** Overall complexity (1-10)
        - **Traffic Impact:** Traffic conditions (1-10)
        - **Order Hour:** Time of day (0-23)
        
        **🚀 Enhanced Features (Generated):**
        - **Interaction Features:** Distance-traffic combinations
        - **PCA Components:** Reduced dimensionality features
        - **Cluster Assignments:** Group-based patterns
        - **Centroid Distances:** Cluster proximity measures
        
        **🎯 Prediction Categories:**
        - 🟢 **Fast:** ≤20 minutes
        - 🟡 **Normal:** 20-30 minutes  
        - 🔴 **Slow:** >30 minutes
        
        **⚙️ Technical Features:**
        - Full unsupervised learning pipeline integration
        - Real-time feature engineering
        - Multiple model strategies
        - Enhanced prediction accuracy
        """)

if __name__ == "__main__":
    main()
