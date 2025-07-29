import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import os
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
# MODEL LOADING AND CHECKING
# =========================================================================
def check_and_load_models():
    """Check and load all required models"""
    
    # Required files
    required_files = {
        'best_model_enhanced.pkl': 'Supervised ML Model',
        'scaler_enhanced.pkl': 'Supervised Scaler',
        'model_metadata_enhanced.pkl': 'Model Metadata',
        'unsupervised_scaler.pkl': 'Feature Engineering Scaler',
        'pca_model.pkl': 'PCA Model',
        'kmeans_model.pkl': 'K-Means Model',
        'feature_info.pkl': 'Feature Information'
    }
    
    models = {}
    missing_files = []
    
    # Check each file
    for filename, description in required_files.items():
        if os.path.exists(filename):
            try:
                if filename.endswith('metadata_enhanced.pkl') or filename == 'feature_info.pkl':
                    with open(filename, 'rb') as f:
                        models[filename] = pickle.load(f)
                else:
                    models[filename] = joblib.load(filename)
                
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {e}")
                missing_files.append(filename)
        else:
            missing_files.append(filename)
    
    return models, missing_files, required_files

# =========================================================================
# FEATURE ENGINEERING
# =========================================================================
def create_all_features(data):
    """Create all interaction features needed"""
    
    # Distance-Traffic interactions
    data['Distance_Traffic_Challenge'] = (
        (data['Distance (km)'] / data['Distance (km)'].max()) * 0.5 + 
        (data['Traffic Impact'] / data['Traffic Impact'].max()) * 0.5
    )
    
    data['Distance_Traffic_Product'] = data['Distance (km)'] * data['Traffic Impact']
    data['Traffic_Per_KM'] = data['Traffic Impact'] / (data['Distance (km)'] + 0.1)
    
    conditions = [
        (data['Distance (km)'] <= 4) & (data['Traffic Impact'] <= 4),
        (data['Distance (km)'] <= 4) & (data['Traffic Impact'] > 4),
        (data['Distance (km)'] > 4) & (data['Traffic Impact'] <= 4),
        (data['Distance (km)'] > 4) & (data['Traffic Impact'] > 4)
    ]
    choices = [1, 2, 3, 4]
    data['Distance_Traffic_Category'] = np.select(conditions, choices, default=3)
    
    data['Delivery_Challenge_Index'] = (
        data['Distance (km)'] * 0.3 + 
        data['Traffic Impact'] * 0.4 + 
        data['Pizza Complexity'] * 0.3
    )
    
    # Pizza profile (both versions)
    data['Pizza_Profile_Score'] = (
        data['Pizza Type'] * 0.3 + 
        data['Topping Density'] * 0.4 + 
        data['Pizza Complexity'] * 0.3
    )
    data['Pizza_Proofile_Score'] = data['Pizza_Profile_Score'].copy()
    
    return data

# =========================================================================
# PREDICTION PIPELINE
# =========================================================================
def make_prediction(input_data, models):
    """Complete prediction pipeline"""
    
    try:
        # Step 1: Feature engineering
        enhanced_data = create_all_features(input_data.copy())
        
        # Step 2: Get feature info
        feature_info = models['feature_info.pkl']
        all_features = feature_info['all_features']
        
        # Step 3: Scale features
        unsupervised_scaler = models['unsupervised_scaler.pkl']
        X_scaled = unsupervised_scaler.transform(enhanced_data[all_features])
        
        # Step 4: PCA
        pca_model = models['pca_model.pkl']
        scaled_features = [f'{f}_scaled' for f in all_features]
        scaled_df = pd.DataFrame(X_scaled, columns=scaled_features)
        
        top_6_features = feature_info['top_6_features']
        X_for_pca = scaled_df[top_6_features].values
        X_pca = pca_model.transform(X_for_pca)
        
        # Add PCA components
        for i in range(X_pca.shape[1]):
            enhanced_data[f'PC{i+1}'] = X_pca[:, i]
        
        # Step 5: Clustering
        kmeans_model = models['kmeans_model.pkl']
        cluster_id = kmeans_model.predict(X_pca)[0]
        enhanced_data['Cluster_ID'] = cluster_id
        
        centroid = kmeans_model.cluster_centers_[cluster_id]
        distance_to_centroid = np.linalg.norm(X_pca[0] - centroid)
        enhanced_data['Distance_to_Centroid'] = distance_to_centroid
        
        cluster_avg_mapping = {0: 22.5, 1: 28.3, 2: 35.1}
        enhanced_data['Cluster_Avg_Delivery'] = cluster_avg_mapping.get(cluster_id, 27.5)
        
        # Step 6: Final prediction
        model = models['best_model_enhanced.pkl']
        scaler = models['scaler_enhanced.pkl']
        metadata = models['model_metadata_enhanced.pkl']
        
        required_features = metadata['features']
        
        # Handle missing features
        for feature in required_features:
            if feature not in enhanced_data.columns:
                if 'PC' in feature:
                    enhanced_data[feature] = [0.0]
                elif feature == 'Cluster_ID':
                    enhanced_data[feature] = [0]
                else:
                    enhanced_data[feature] = [1.0]
        
        X_final = enhanced_data[required_features]
        
        if metadata.get('scaled_data', False):
            X_final_scaled = scaler.transform(X_final)
            prediction = model.predict(X_final_scaled)[0]
        else:
            prediction = model.predict(X_final)[0]
        
        pipeline_info = {
            'cluster_id': cluster_id,
            'distance_to_centroid': distance_to_centroid,
            'features_created': len(enhanced_data.columns)
        }
        
        return prediction, pipeline_info, None
        
    except Exception as e:
        return None, None, str(e)

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def create_gauge_chart(prediction):
    """Create prediction gauge chart"""
    
    if prediction <= 20:
        color = "green"
        category = "Fast"
    elif prediction <= 30:
        color = "orange"
        category = "Normal"
    else:
        color = "red"
        category = "Slow"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create gauge background
    theta = np.linspace(0, np.pi, 100)
    ax.fill_between(theta, 0, 1, where=(theta <= np.pi * 0.4), alpha=0.3, color='green')
    ax.fill_between(theta, 0, 1, where=(theta > np.pi * 0.4) & (theta <= np.pi * 0.6), alpha=0.3, color='orange')
    ax.fill_between(theta, 0, 1, where=(theta > np.pi * 0.6), alpha=0.3, color='red')
    
    # Needle
    needle_angle = np.pi * (1 - min(prediction / 50, 1))
    ax.plot([needle_angle, needle_angle], [0, 0.8], color='black', linewidth=3)
    
    # Text
    ax.text(np.pi/2, 0.5, f'{prediction:.1f}\nminutes', ha='center', va='center', 
            fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    ax.text(np.pi/2, 0.2, category, ha='center', va='center', 
            fontsize=14, fontweight='bold', color=color)
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Predicted Delivery Time', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def generate_demo_models():
    """Generate demo models for testing"""
    
    try:
        from sklearn.preprocessing import RobustScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestRegressor
        
        # Create demo data
        np.random.seed(42)
        n_samples = 300
        
        data = pd.DataFrame({
            'Pizza Type': np.random.randint(1, 6, n_samples),
            'Distance (km)': np.random.uniform(0.5, 15, n_samples),
            'Is Weekend': np.random.randint(0, 2, n_samples),
            'Topping Density': np.random.randint(1, 11, n_samples),
            'Order Month': np.random.randint(1, 13, n_samples),
            'Pizza Complexity': np.random.randint(1, 11, n_samples),
            'Traffic Impact': np.random.randint(1, 11, n_samples),
            'Order Hour': np.random.randint(0, 24, n_samples),
        })
        
        # Feature engineering
        data = create_all_features(data)
        
        features = ['Pizza Type', 'Distance (km)', 'Is Weekend', 'Topping Density', 
                   'Order Month', 'Pizza Complexity', 'Traffic Impact', 'Order Hour',
                   'Distance_Traffic_Challenge', 'Distance_Traffic_Product', 
                   'Traffic_Per_KM', 'Distance_Traffic_Category', 
                   'Delivery_Challenge_Index', 'Pizza_Profile_Score', 'Pizza_Proofile_Score']
        
        # Create target variable
        target = (data['Distance (km)'] * 1.5 + 
                 data['Traffic Impact'] * 2 + 
                 data['Pizza Complexity'] * 1.2 + 
                 np.random.normal(0, 3, n_samples) + 20)
        
        # Create models
        # 1. Unsupervised scaler
        scaler = RobustScaler()
        scaler.fit(data[features])
        joblib.dump(scaler, 'unsupervised_scaler.pkl')
        
        # 2. PCA
        X_scaled = scaler.transform(data[features])
        pca = PCA(n_components=4)
        pca.fit(X_scaled[:, :6])
        joblib.dump(pca, 'pca_model.pkl')
        
        # 3. KMeans
        X_pca = pca.transform(X_scaled[:, :6])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        joblib.dump(kmeans, 'kmeans_model.pkl')
        
        # 4. Add enhanced features
        for i in range(4):
            data[f'PC{i+1}'] = X_pca[:, i]
        
        data['Cluster_ID'] = kmeans.labels_
        data['Distance_to_Centroid'] = [np.linalg.norm(X_pca[i] - kmeans.cluster_centers_[kmeans.labels_[i]]) 
                                       for i in range(len(X_pca))]
        data['Cluster_Avg_Delivery'] = 27.5
        
        enhanced_features = features + ['PC1', 'PC2', 'PC3', 'PC4', 'Cluster_ID', 'Distance_to_Centroid', 'Cluster_Avg_Delivery']
        
        # 5. Supervised model
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(data[enhanced_features], target, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, 'best_model_enhanced.pkl')
        
        # 6. Supervised scaler
        sup_scaler = RobustScaler()
        sup_scaler.fit(X_train)
        joblib.dump(sup_scaler, 'scaler_enhanced.pkl')
        
        # 7. Metadata
        from sklearn.metrics import r2_score, mean_squared_error
        y_pred = model.predict(X_test)
        
        metadata = {
            'model_name': 'Random Forest',
            'strategy': 'All_Enhanced',
            'features': enhanced_features,
            'feature_types': {
                'original': features[:8],
                'cluster': ['Cluster_ID', 'Distance_to_Centroid', 'Cluster_Avg_Delivery'],
                'pca': ['PC1', 'PC2', 'PC3', 'PC4']
            },
            'scaled_data': False,
            'performance': {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        }
        
        with open('model_metadata_enhanced.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # 8. Feature info
        feature_info = {
            'original_features': features[:8],
            'interaction_features': features[8:],
            'all_features': features,
            'top_6_features': [f'{f}_scaled' for f in features[:6]],
            'target': 'Delivery Duration (min)'
        }
        
        with open('feature_info.pkl', 'wb') as f:
            pickle.dump(feature_info, f)
        
        return True, "Demo models created successfully!"
        
    except Exception as e:
        return False, f"Error creating demo models: {str(e)}"

# =========================================================================
# MAIN APPLICATION
# =========================================================================
def main():
    
    # Header
    st.title("🍕 Pizza Delivery Time Predictor")
    st.markdown("### AI-Powered Delivery Time Estimation")
    
    # Load models
    models, missing_files, required_files = check_and_load_models()
    
    # Check if all models are loaded
    if missing_files:
        st.error(f"❌ **Missing {len(missing_files)} required files:**")
        
        for filename in missing_files:
            description = required_files[filename]
            st.write(f"   📄 {filename} - {description}")
        
        st.markdown("---")
        
        if st.button("🚀 **Generate Demo Models**", type="primary"):
            with st.spinner("Generating models... This may take a moment."):
                success, message = generate_demo_models()
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("🔄 **Please refresh the page** to load the generated models.")
                    if st.button("🔄 Refresh Page"):
                        st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        st.info("💡 **Alternative:** Place the required .pkl files in the same directory as this app.")
        
        return
    
    st.success("✅ **All models loaded successfully!**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        metadata = models['model_metadata_enhanced.pkl']
        st.write(f"**Model:** {metadata['model_name']}")
        st.write(f"**Strategy:** {metadata['strategy']}")
        st.write(f"**Features:** {len(metadata['features'])}")
        st.write(f"**R² Score:** {metadata['performance']['r2_score']:.4f}")
        st.write(f"**RMSE:** {metadata['performance']['rmse']:.2f} min")
        
        st.markdown("---")
        st.write("**Pipeline:**")
        st.write("✅ Feature Engineering")
        st.write("✅ RobustScaler") 
        st.write("✅ PCA Transform")
        st.write("✅ K-Means Clustering")
        st.write("✅ ML Prediction")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Order Details")
        
        with st.form("prediction_form"):
            
            # Input fields
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                pizza_type = st.selectbox("🍕 Pizza Type", [1,2,3,4,5], index=2, help="1=Basic, 5=Premium")
                distance = st.slider("📍 Distance (km)", 0.5, 15.0, 5.0, 0.5)
                topping_density = st.slider("🧀 Topping Density", 1, 10, 5, help="1=Light, 10=Heavy")
                is_weekend = st.selectbox("📅 Weekend?", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with input_col2:
                order_month = st.selectbox("📆 Month", list(range(1,13)), index=5)
                pizza_complexity = st.slider("⚙️ Complexity", 1, 10, 5, help="1=Simple, 10=Complex")
                traffic_impact = st.slider("🚦 Traffic", 1, 10, 5, help="1=Clear, 10=Heavy")
                order_hour = st.slider("🕐 Hour", 0, 23, 18)
            
            # Submit button
            submitted = st.form_submit_button("🎯 **Predict Delivery Time**", type="primary", use_container_width=True)
            
            if submitted:
                # Prepare input
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
                
                # Make prediction
                with st.spinner("🔄 Processing prediction..."):
                    prediction, pipeline_info, error = make_prediction(input_data, models)
                
                if error:
                    st.error(f"❌ Prediction failed: {error}")
                else:
                    st.success("✅ **Prediction completed!**")
                    
                    # Store results
                    st.session_state.prediction = prediction
                    st.session_state.pipeline_info = pipeline_info
                    st.session_state.input_summary = {
                        'Pizza Type': f"{pizza_type}/5",
                        'Distance': f"{distance} km",
                        'Traffic': f"{traffic_impact}/10",
                        'Complexity': f"{pizza_complexity}/10",
                        'Weekend': "Yes" if is_weekend else "No",
                        'Hour': f"{order_hour}:00"
                    }
    
    with col2:
        st.header("📊 Prediction")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            pipeline_info = st.session_state.pipeline_info
            
            # Gauge chart
            gauge_fig = create_gauge_chart(prediction)
            st.pyplot(gauge_fig, clear_figure=True)
            
            # Results
            st.subheader("📋 Results")
            st.write(f"**Estimated Time:** {prediction:.1f} minutes")
            
            if pipeline_info:
                st.write(f"**Cluster:** {pipeline_info['cluster_id']}")
                st.write(f"**Features Used:** {pipeline_info['features_created']}")
            
            # Input summary
            st.subheader("📝 Order Summary")
            for key, value in st.session_state.input_summary.items():
                st.write(f"**{key}:** {value}")
            
            # Category
            if prediction <= 20:
                st.success("🟢 **Fast Delivery**")
            elif prediction <= 30:
                st.warning("🟡 **Normal Delivery**")
            else:
                st.error("🔴 **Slow Delivery**")
        
        else:
            st.info("👆 **Enter order details** and click predict to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("**🍕 Pizza Delivery Time Predictor** | Powered by Machine Learning")

if __name__ == "__main__":
    main()
