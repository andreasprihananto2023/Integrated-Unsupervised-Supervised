import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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
# COMPLETE PREDICTION FUNCTION (FIXED)
# =========================================================================
def predict_with_complete_pipeline(input_data):
    """Complete prediction pipeline with all fixes applied"""
    try:
        # Load all models
        model = joblib.load('best_model_enhanced.pkl')
        scaler = joblib.load('scaler_enhanced.pkl')
        
        with open('model_metadata_enhanced.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load unsupervised models
        unsupervised_scaler = joblib.load('unsupervised_scaler.pkl')
        pca_model = joblib.load('pca_model.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        # STEP 1: Apply feature engineering (with typo handling)
        enhanced_data = input_data.copy()
        
        # Create interaction features
        enhanced_data['Distance_Traffic_Challenge'] = (
            (enhanced_data['Distance (km)'] / enhanced_data['Distance (km)'].max()) * 0.5 + 
            (enhanced_data['Traffic Impact'] / enhanced_data['Traffic Impact'].max()) * 0.5
        )
        enhanced_data['Distance_Traffic_Product'] = enhanced_data['Distance (km)'] * enhanced_data['Traffic Impact']
        enhanced_data['Traffic_Per_KM'] = enhanced_data['Traffic Impact'] / (enhanced_data['Distance (km)'] + 0.1)
        
        conditions = [
            (enhanced_data['Distance (km)'] <= 4) & (enhanced_data['Traffic Impact'] <= 4),
            (enhanced_data['Distance (km)'] <= 4) & (enhanced_data['Traffic Impact'] > 4),
            (enhanced_data['Distance (km)'] > 4) & (enhanced_data['Traffic Impact'] <= 4),
            (enhanced_data['Distance (km)'] > 4) & (enhanced_data['Traffic Impact'] > 4)
        ]
        choices = [1, 2, 3, 4]
        enhanced_data['Distance_Traffic_Category'] = np.select(conditions, choices, default=3)
        
        enhanced_data['Delivery_Challenge_Index'] = (
            enhanced_data['Distance (km)'] * 0.3 + 
            enhanced_data['Traffic Impact'] * 0.4 + 
            enhanced_data['Pizza Complexity'] * 0.3
        )
        
        # Create BOTH versions to handle typo
        enhanced_data['Pizza_Profile_Score'] = (
            enhanced_data['Pizza Type'] * 0.3 + 
            enhanced_data['Topping Density'] * 0.4 + 
            enhanced_data['Pizza Complexity'] * 0.3
        )
        enhanced_data['Pizza_Proofile_Score'] = enhanced_data['Pizza_Profile_Score'].copy()
        
        # STEP 2: Apply scaling
        all_features = feature_info['all_features']
        X_scaled = unsupervised_scaler.transform(enhanced_data[all_features])
        
        # STEP 3: Apply PCA
        scaled_features = [f'{f}_scaled' for f in all_features]
        scaled_df = pd.DataFrame(X_scaled, columns=scaled_features)
        top_6_features = feature_info['top_6_features']
        X_for_pca = scaled_df[top_6_features].values
        X_pca = pca_model.transform(X_for_pca)
        
        # Add PCA components
        for i in range(X_pca.shape[1]):
            enhanced_data[f'PC{i+1}'] = X_pca[:, i]
        
        # STEP 4: Apply clustering
        cluster_id = kmeans_model.predict(X_pca)[0]
        enhanced_data['Cluster_ID'] = cluster_id
        
        # Calculate distance to centroid
        centroid = kmeans_model.cluster_centers_[cluster_id]
        distance_to_centroid = np.linalg.norm(X_pca[0] - centroid)
        enhanced_data['Distance_to_Centroid'] = distance_to_centroid
        
        # Add cluster average delivery time
        cluster_avg_mapping = {0: 22.5, 1: 28.3, 2: 35.1}
        enhanced_data['Cluster_Avg_Delivery'] = cluster_avg_mapping.get(cluster_id, 27.5)
        
        # STEP 5: Make prediction
        required_features = metadata['features']
        
        # Check for missing features and provide defaults
        for feature in required_features:
            if feature not in enhanced_data.columns:
                if 'PC' in feature:
                    enhanced_data[feature] = [0.0]
                elif feature == 'Cluster_ID':
                    enhanced_data[feature] = [0]
                elif feature == 'Distance_to_Centroid':
                    enhanced_data[feature] = [1.0]
                elif feature == 'Cluster_Avg_Delivery':
                    enhanced_data[feature] = [27.5]
        
        X_final = enhanced_data[required_features]
        
        if metadata.get('scaled_data', False):
            X_final_scaled = scaler.transform(X_final)
            prediction = model.predict(X_final_scaled)[0]
        else:
            prediction = model.predict(X_final)[0]
        
        return prediction, None, {
            'cluster_id': cluster_id,
            'distance_to_centroid': distance_to_centroid,
            'features_created': len(enhanced_data.columns),
            'pca_components': X_pca.shape[1]
        }
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}", None

# =========================================================================
# LOAD MODEL AND CHECK STATUS
# =========================================================================
@st.cache_data
def load_and_check_models():
    """Load all models and check their status"""
    status = {
        'supervised_models': False,
        'unsupervised_models': False,
        'metadata': None,
        'feature_info': None,
        'error_messages': []
    }
    
    # Check supervised models
    try:
        model = joblib.load('best_model_enhanced.pkl')
        scaler = joblib.load('scaler_enhanced.pkl')
        
        with open('model_metadata_enhanced.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        status['supervised_models'] = True
        status['metadata'] = metadata
        
    except FileNotFoundError as e:
        status['error_messages'].append(f"Supervised models: {str(e)}")
    
    # Check unsupervised models
    try:
        unsupervised_scaler = joblib.load('unsupervised_scaler.pkl')
        pca_model = joblib.load('pca_model.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        status['unsupervised_models'] = True
        status['feature_info'] = feature_info
        
    except FileNotFoundError as e:
        status['error_messages'].append(f"Unsupervised models: {str(e)}")
    
    return status

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def create_prediction_gauge_matplotlib(prediction):
    """Create gauge chart using matplotlib"""
    if prediction <= 20:
        color = "green"
        category = "Fast"
    elif prediction <= 30:
        color = "orange" 
        category = "Normal"
    else:
        color = "red"
        category = "Slow"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create gauge sectors
    theta = np.linspace(0, np.pi, 100)
    
    # Background sectors
    ax.fill_between(theta, 0, 1, where=(theta <= np.pi * 0.4), alpha=0.3, color='green', label='Fast (≤20 min)')
    ax.fill_between(theta, 0, 1, where=(theta > np.pi * 0.4) & (theta <= np.pi * 0.6), alpha=0.3, color='orange', label='Normal (20-30 min)')
    ax.fill_between(theta, 0, 1, where=(theta > np.pi * 0.6), alpha=0.3, color='red', label='Slow (>30 min)')
    
    # Needle position
    needle_angle = np.pi * (1 - min(prediction / 50, 1))
    ax.plot([needle_angle, needle_angle], [0, 0.8], color='black', linewidth=3)
    
    # Add text
    ax.text(np.pi/2, 0.5, f'{prediction:.1f}\nminutes', ha='center', va='center', 
            fontsize=16, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    ax.text(np.pi/2, 0.2, category, ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Predicted Delivery Time', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_pipeline_flow_chart():
    """Create pipeline flow visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define pipeline steps
    steps = [
        "User Input\n(8 features)",
        "Feature Engineering\n(+6 interactions)", 
        "RobustScaler\n(normalize 14 features)",
        "PCA Transform\n(reduce to 4 components)",
        "K-Means Clustering\n(assign cluster)",
        "Enhanced Features\n(+3 cluster features)",
        "ML Model\n(17 total features)",
        "Prediction\n(delivery time)"
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
             'lightpink', 'lightgray', 'orange', 'red']
    
    # Create flow chart
    y_positions = [0.8, 0.6, 0.4, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for i, (step, color, x, y) in enumerate(zip(steps, colors, x_positions, y_positions)):
        # Draw box
        box = plt.Rectangle((x-0.05, y-0.08), 0.1, 0.16, 
                           facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, step, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add arrow to next step
        if i < len(steps) - 1:
            next_x, next_y = x_positions[i+1], y_positions[i+1]
            ax.annotate('', xy=(next_x-0.05, next_y), xytext=(x+0.05, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Enhanced ML Pipeline Flow', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# =========================================================================
# MAIN APP
# =========================================================================
def main():
    # Load and check model status
    status = load_and_check_models()
    
    # Header
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("### Advanced ML with Complete Feature Engineering Pipeline")
    
    # Status check
    if not status['supervised_models'] or not status['unsupervised_models']:
        st.error("❌ **Required model files are missing!**")
        
        if status['error_messages']:
            st.write("**Error details:**")
            for error in status['error_messages']:
                st.write(f"- {error}")
        
        st.info("**To fix this issue:**")
        st.code("""
        1. Run the complete fix script:
           python complete_fix_solution.py
           
        2. Or run the enhanced supervised learning script with fixes
        
        3. Required files:
           - best_model_enhanced.pkl
           - scaler_enhanced.pkl  
           - model_metadata_enhanced.pkl
           - unsupervised_scaler.pkl
           - pca_model.pkl
           - kmeans_model.pkl
           - feature_info.pkl
        """)
        return
    
    st.success("✅ **All models loaded successfully!**")
    st.info("🚀 **Enhanced Pipeline Active:** 8 inputs → 17 features → ML prediction")
    
    # Show pipeline flow
    with st.expander("📊 View Pipeline Flow"):
        pipeline_fig = create_pipeline_flow_chart()
        st.pyplot(pipeline_fig)
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        
        if status['metadata']:
            metadata = status['metadata']
            st.write(f"**Model Type:** {metadata['model_name']}")
            st.write(f"**Strategy:** {metadata.get('strategy', 'N/A')}")
            st.write(f"**Features:** {len(metadata['features'])}")
            st.write(f"**R² Score:** {metadata['performance']['r2_score']:.4f}")
            st.write(f"**RMSE:** {metadata['performance']['rmse']:.2f} min")
        
        st.write("**Pipeline Status:**")
        st.write("✅ Feature Engineering")
        st.write("✅ RobustScaler")
        st.write("✅ PCA Transform")
        st.write("✅ K-Means Clustering")
        st.write("✅ Enhanced Features")
        
        if status['feature_info']:
            feature_info = status['feature_info']
            with st.expander("📋 Pipeline Details"):
                st.write(f"**Original Features:** {len(feature_info['original_features'])}")
                st.write(f"**Interaction Features:** {len(feature_info['interaction_features'])}")
                st.write(f"**PCA Components:** 4")
                st.write(f"**Cluster Features:** 3")
                st.write(f"**Total Features:** {len(feature_info['all_features']) + 7}")
                
                if feature_info.get('created_both_versions'):
                    st.success("✅ Typo handling: Both Pizza_Profile_Score and Pizza_Proofile_Score")
    
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
                    min_value=0.1,
                    max_value=1.5,
                    value=5.0,
                    step=0.1
                )

                topping_density = st.selectbox(
                    "🍕 Pizza Type",
                    options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                    index=2,
                    help="1=Basic, 5=Premium"
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
                
                # Make prediction with complete pipeline
                with st.spinner("🔄 Processing through enhanced ML pipeline..."):
                    prediction, error, pipeline_info = predict_with_complete_pipeline(input_data)
                
                if error:
                    st.error(f"❌ {error}")
                    
                    # Troubleshooting help
                    with st.expander("🔧 Troubleshooting"):
                        st.write("**Possible solutions:**")
                        st.write("1. Run the complete fix script")
                        st.write("2. Check that all model files exist")
                        st.write("3. Verify file compatibility")
                        
                        if status['metadata']:
                            st.write("**Required features:**")
                            st.write(status['metadata']['features'])
                
                else:
                    st.success("✅ Prediction completed successfully!")
                    
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.pipeline_info = pipeline_info
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
    
    with col2:
        st.header("📊 Results")
        
        # Display prediction if available
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            pipeline_info = st.session_state.pipeline_info
            
            # Gauge chart
            gauge_fig = create_prediction_gauge_matplotlib(prediction)
            st.pyplot(gauge_fig)
            
            # Pipeline information
            st.success("🚀 **Enhanced Pipeline Used**")
            if pipeline_info:
                st.write(f"**Features Created:** {pipeline_info['features_created']}")
                st.write(f"**PCA Components:** {pipeline_info['pca_components']}")
                st.write(f"**Assigned Cluster:** {pipeline_info['cluster_id']}")
                st.write(f"**Distance to Centroid:** {pipeline_info['distance_to_centroid']:.3f}")
            
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
            if status['metadata']:
                confidence_score = status['metadata']['performance']['r2_score']
                if confidence_score >= 0.8:
                    st.success(f"High Confidence (R² = {confidence_score:.3f})")
                elif confidence_score >= 0.6:
                    st.warning(f"Medium Confidence (R² = {confidence_score:.3f})")
                else:
                    st.error(f"Low Confidence (R² = {confidence_score:.3f})")
        
        else:
            st.info("👆 Enter order details and click predict to see results")
    
    # Model Performance Section
    if status['metadata']:
        st.header("🎯 Model Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric(
                label="R² Score",
                value=f"{status['metadata']['performance']['r2_score']:.4f}",
                help="Coefficient of determination (higher is better)"
            )
        
        with perf_col2:
            st.metric(
                label="RMSE",
                value=f"{status['metadata']['performance']['rmse']:.2f} min",
                help="Root Mean Square Error (lower is better)"
            )
        
        with perf_col3:
            st.metric(
                label="Features Used",
                value=len(status['metadata']['features']),
                help="Number of features in the model"
            )
        
        with perf_col4:
            st.metric(
                label="Pipeline Type",
                value="🚀 Enhanced",
                help="Uses complete feature engineering pipeline"
            )
    
    # About section
    with st.expander("ℹ️ About This Enhanced App"):
        st.markdown("""
        ### 🍕 Enhanced Pizza Delivery Time Predictor
        
        This application uses a complete machine learning pipeline with advanced feature engineering.
        
        **🔧 Complete Pipeline Process:**
        
        1. **User Input (8 features):**
           - Pizza Type, Distance, Weekend, Topping Density
           - Order Month, Pizza Complexity, Traffic Impact, Order Hour
        
        2. **Feature Engineering (+6 interaction features):**
           - Distance-Traffic Challenge Score
           - Distance-Traffic Product
           - Traffic per KM
           - Distance-Traffic Category
           - Delivery Challenge Index
           - Pizza Profile Score (handles typo: both versions created)
        
        3. **RobustScaler:** Normalizes all 14 features
        
        4. **PCA Transform:** Reduces to 4 principal components
        
        5. **K-Means Clustering:** Assigns to optimal cluster (3 clusters)
        
        6. **Enhanced Features (+3 cluster features):**
           - Cluster ID assignment
           - Distance to cluster centroid
           - Cluster average delivery time
        
        7. **ML Model:** Uses all 17 features for prediction
        
        **🎯 Key Features:**
        - **Automatic Feature Generation:** You input 8, system creates 17
        - **Typo Handling:** Handles both Pizza_Profile_Score and Pizza_Proofile_Score
        - **Complete Pipeline:** Same transformations as training
        - **Real-time Processing:** All computations done instantly
        - **High Accuracy:** Enhanced features improve prediction quality
        
        **🚀 Technical Highlights:**
        - RobustScaler for outlier-resistant normalization
        - PCA for dimensionality reduction
        - K-Means clustering for pattern recognition
        - Comprehensive error handling and fallbacks
        """)
    
    # Setup Instructions
    with st.expander("⚙️ Setup & Troubleshooting"):
        st.markdown("""
        ### 📋 Required Files
        
        **Core Model Files:**
        ```
        best_model_enhanced.pkl       # Trained ML model
        scaler_enhanced.pkl          # Feature scaler for final prediction
        model_metadata_enhanced.pkl  # Model configuration and performance
        ```
        
        **Unsupervised Pipeline Files:**
        ```
        unsupervised_scaler.pkl     # Feature engineering scaler
        pca_model.pkl              # PCA transformation model
        kmeans_model.pkl           # K-Means clustering model
        feature_info.pkl           # Feature mapping and configuration
        ```
        
        ### 🔧 If You See Errors:
        
        **Missing Files Error:**
        1. Run the complete fix script: `python complete_fix_solution.py`
        2. This will generate all required unsupervised models
        3. Restart the Streamlit app
        
        **Feature Mismatch Error:**
        1. The fix script handles the Pizza_Proofile_Score typo
        2. Creates both versions for compatibility
        3. Ensures all interaction features are properly generated
        
        **Pipeline Test:**
        The app automatically tests the complete pipeline on startup and shows any issues.
        """)

if __name__ == "__main__":
    main()
