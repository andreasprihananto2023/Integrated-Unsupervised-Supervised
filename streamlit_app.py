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
        
        return model, scaler, metadata, correlations, "enhanced"
        
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
            
            return model, scaler, metadata, correlations, "original"
            
        except FileNotFoundError:
            return None, None, None, None, None

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def predict_delivery_time(input_data, model, scaler, metadata):
    """Make prediction using loaded model"""
    try:
        # Prepare features according to model requirements
        required_features = metadata['features']
        
        # Check if we have all required features
        missing_features = [f for f in required_features if f not in input_data.columns]
        if missing_features:
            return None, f"Missing features: {missing_features}"
        
        # Select required features
        X = input_data[required_features]
        
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
                'Feature': feature,
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
            'PCA': '#2ca02c'
        }
    )
    fig.update_layout(height=400, showlegend=True)
    
    return fig

def get_feature_type(feature, metadata):
    """Determine feature type"""
    feature_types = metadata.get('feature_types', {})
    
    if feature in feature_types.get('original', []):
        return 'Original'
    elif feature in feature_types.get('cluster', []):
        return 'Cluster'
    elif feature in feature_types.get('pca', []):
        return 'PCA'
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

# =========================================================================
# MAIN APP
# =========================================================================
def main():
    # Load model artifacts
    model, scaler, metadata, correlations, model_type = load_model_artifacts()
    
    # Header
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("### Predict delivery time using machine learning")
    
    if model is None:
        st.error("❌ **Model files not found!**")
        st.info("Please ensure the following files are in the same directory:")
        st.code("""
        - best_model_enhanced.pkl (or best_model.pkl)
        - scaler_enhanced.pkl (or scaler.pkl) 
        - model_metadata_enhanced.pkl (or model_metadata.pkl)
        """)
        return
    
    # Model info
    st.success(f"✅ **{model_type.title()} model loaded successfully!**")
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model Type:** {metadata['model_name']}")
        st.write(f"**Strategy:** {metadata.get('strategy', 'N/A')}")
        st.write(f"**Features:** {len(metadata['features'])}")
        st.write(f"**R² Score:** {metadata['performance']['r2_score']:.4f}")
        st.write(f"**RMSE:** {metadata['performance']['rmse']:.2f} min")
        
        if metadata.get('enhancement_used', False):
            st.write("**Enhanced Features:** ✅ Used")
        else:
            st.write("**Enhanced Features:** ❌ Not used")
        
        # Feature list
        with st.expander("📋 Features Used"):
            for i, feature in enumerate(metadata['features'], 1):
                feature_type = get_feature_type(feature, metadata)
                emoji = "🔹" if feature_type == "Original" else "🔸" if feature_type == "Cluster" else "🔺"
                st.write(f"{emoji} {feature}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Make Prediction")
        
        # Check if we need enhanced features
        needs_enhanced = any(f in metadata['features'] for f in ['Cluster_ID', 'PC1', 'Distance_to_Centroid'])
        
        if needs_enhanced:
            st.warning("⚠️ This model requires enhanced features from unsupervised analysis. For demonstration, simplified input is provided.")
        
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
                
                # Add enhanced features if needed (simplified for demo)
                if needs_enhanced:
                    # Add dummy enhanced features for demonstration
                    if 'Cluster_ID' in metadata['features']:
                        # Simple clustering logic based on distance and traffic
                        if distance <= 4 and traffic_impact <= 4:
                            cluster_id = 0
                        elif distance <= 4 and traffic_impact > 4:
                            cluster_id = 1
                        else:
                            cluster_id = 2
                        input_data['Cluster_ID'] = [cluster_id]
                    
                    # Add PCA components (simplified)
                    for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
                        if pc in metadata['features']:
                            input_data[pc] = [np.random.normal(0, 1)]  # Random for demo
                    
                    # Add other enhanced features
                    if 'Distance_to_Centroid' in metadata['features']:
                        input_data['Distance_to_Centroid'] = [np.random.uniform(0.5, 2.0)]
                    
                    if 'Cluster_Avg_Delivery' in metadata['features']:
                        input_data['Cluster_Avg_Delivery'] = [25 + np.random.normal(0, 5)]
                
                # Make prediction
                prediction, error = predict_delivery_time(input_data, model, scaler, metadata)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Prediction completed!")
                    
                    # Store prediction in session state for gauge chart
                    st.session_state.prediction = prediction
                    st.session_state.input_summary = {
                        'Distance': f"{distance} km",
                        'Traffic': f"{traffic_impact}/10",
                        'Pizza Type': f"{pizza_type}/5",
                        'Weekend': "Yes" if is_weekend else "No",
                        'Hour': f"{order_hour}:00"
                    }
    
    with col2:
        st.header("📊 Results")
        
        # Display prediction if available
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            
            # Gauge chart
            gauge_fig = create_prediction_gauge(prediction)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
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
        enhancement_status = "✅ Enhanced" if metadata.get('enhancement_used', False) else "📊 Standard"
        st.metric(
            label="Model Type",
            value=enhancement_status,
            help="Whether enhanced features from clustering/PCA are used"
        )
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### 🍕 Pizza Delivery Time Predictor
        
        This application uses machine learning to predict pizza delivery times based on various factors:
        
        **📊 Input Features:**
        - **Pizza Type:** Complexity level of the pizza (1-5)
        - **Distance:** Delivery distance in kilometers
        - **Weekend:** Whether the order is on weekend
        - **Topping Density:** Amount of toppings (1-10)
        - **Order Month:** Month of the year
        - **Pizza Complexity:** Overall complexity score
        - **Traffic Impact:** Current traffic conditions (1-10)
        - **Order Hour:** Time of day (0-23)
        
        **🤖 Model Features:**
        - Trained on pizza delivery data
        - Uses advanced feature engineering
        - Incorporates clustering and PCA insights
        - Provides accurate time predictions
        
        **🎯 Prediction Categories:**
        - 🟢 **Fast:** ≤20 minutes
        - 🟡 **Normal:** 20-30 minutes  
        - 🔴 **Slow:** >30 minutes
        """)

if __name__ == "__main__":
    main()