
def predict_with_complete_pipeline(input_data):
    """Complete prediction pipeline for Streamlit integration"""
    import pandas as pd
    import numpy as np
    import joblib
    import pickle
    
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
        
        return prediction, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"
