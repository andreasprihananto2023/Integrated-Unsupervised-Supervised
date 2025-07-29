import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =========================================================================
# FEATURE ENGINEERING: DISTANCE-TRAFFIC INTERACTION
# =========================================================================
def add_distance_traffic_interaction(data):
    """Add distance-traffic interaction features"""
    print("\n🔧 FEATURE ENGINEERING: Distance-Traffic Interaction")
    print("-" * 50)
    
    # 1. Distance-Traffic Combined Challenge Score
    # Normalized score combining both factors
    data['Distance_Traffic_Challenge'] = (
        (data['Distance (km)'] / data['Distance (km)'].max()) * 0.5 + 
        (data['Traffic Impact'] / data['Traffic Impact'].max()) * 0.5
    )
    
    # 2. Distance-Traffic Multiplication (Raw Interaction)
    data['Distance_Traffic_Product'] = data['Distance (km)'] * data['Traffic Impact']
    
    # 3. Traffic Impact per KM
    # How much traffic affects each kilometer
    data['Traffic_Per_KM'] = data['Traffic Impact'] / (data['Distance (km)'] + 0.1)  # +0.1 to avoid division by zero
    
    # 4. Distance-Traffic Category (Categorical Interaction)
    # Create categories based on both distance and traffic
    conditions = [
        (data['Distance (km)'] <= 4) & (data['Traffic Impact'] <= 4),           # Low-Low
        (data['Distance (km)'] <= 4) & (data['Traffic Impact'] > 4),            # Low-High  
        (data['Distance (km)'] > 4) & (data['Traffic Impact'] <= 4),            # High-Low
        (data['Distance (km)'] > 4) & (data['Traffic Impact'] > 4)              # High-High
    ]
    choices = [1, 2, 3, 4]  # Low-Low, Low-High, High-Low, High-High
    data['Distance_Traffic_Category'] = np.select(conditions, choices, default=3)
    
    # 5. Delivery Challenge Index
    # Comprehensive challenge score (distance + traffic + complexity)
    data['Delivery_Challenge_Index'] = (
        data['Distance (km)'] * 0.3 + 
        data['Traffic Impact'] * 0.4 + 
        data['Pizza Complexity'] * 0.3
    )
    
    # 6. Pizza Profile Score
    # Weighted combination of pizza characteristics
    data['Pizza_Profile_Score'] = (
        data['Pizza Type'] * 0.3 + 
        data['Topping Density'] * 0.4 + 
        data['Pizza Complexity'] * 0.3
    )
    
    print("✅ Added Distance-Traffic Interaction Features:")
    print("   • Distance_Traffic_Challenge (normalized 0-1)")
    print("   • Distance_Traffic_Product (raw multiplication)")  
    print("   • Traffic_Per_KM (traffic impact per kilometer)")
    print("   • Distance_Traffic_Category (1=Low-Low to 4=High-High)")
    print("   • Delivery_Challenge_Index (comprehensive score)")
    print("   • Pizza_Profile_Score (weighted pizza characteristics)")
    
    # Display feature statistics
    print(f"\n📊 Feature Statistics:")
    interaction_features = ['Distance_Traffic_Challenge', 'Distance_Traffic_Product', 
                           'Traffic_Per_KM', 'Distance_Traffic_Category', 'Delivery_Challenge_Index',
                           'Pizza_Profile_Score']
    
    for feature in interaction_features:
        if feature in data.columns:
            print(f"   {feature:25}: min={data[feature].min():6.2f}, max={data[feature].max():6.2f}, mean={data[feature].mean():6.2f}")
    
    return data

# =========================================================================
# STEP 1: LOAD AND PREPARE DATA (ENHANCED)
# =========================================================================
def load_and_prepare_data():
    """Load data and prepare features (with Distance-Traffic Interaction)"""
    print("\n📊 STEP 1: DATA PREPARATION (WITH INTERACTION FEATURES)")
    print("-" * 60)
    
    # Load data
    data = pd.read_excel('Train Data.xlsx')
    data.columns = data.columns.str.strip()
    
    # Apply feature engineering BEFORE feature selection
    data = add_distance_traffic_interaction(data)
    
    # Define features (ORIGINAL + NEW INTERACTION FEATURES)
    original_features = ['Pizza Type', 'Distance (km)', 'Is Weekend', 'Topping Density', 
                        'Order Month', 'Pizza Complexity', 'Traffic Impact', 'Order Hour']
    
    interaction_features = ['Distance_Traffic_Challenge', 'Distance_Traffic_Product', 
                           'Traffic_Per_KM', 'Distance_Traffic_Category', 'Delivery_Challenge_Index',
                           'Pizza_Profile_Score']
    
    features = original_features + interaction_features
    target = 'Delivery Duration (min)'
    
    # Clean data
    data_clean = data[features + [target]].dropna()
    
    print(f"✅ Data loaded: {len(data_clean)} rows")
    print(f"✅ Original features: {len(original_features)}")
    print(f"✅ Interaction features: {len(interaction_features)}")
    print(f"✅ Total features: {len(features)} (was 8, now {len(features)})")
    print(f"🎯 Target: {target}")
    
    print(f"\n🔧 Feature List:")
    print(f"   Original ({len(original_features)}): {', '.join(original_features)}")
    print(f"   New ({len(interaction_features)}): {', '.join(interaction_features)}")
    
    return data_clean, features, target

# =========================================================================
# STEP 2: ROBUST SCALING
# =========================================================================
def apply_scaling(data_clean, features):
    """Apply RobustScaler to features"""
    print("\n🔧 STEP 2: ROBUST SCALING")
    print("-" * 40)
    
    # Apply RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(data_clean[features])
    
    # Create DataFrame with scaled features
    data_scaled = data_clean.copy()
    for i, feature in enumerate(features):
        data_scaled[f'{feature}_scaled'] = X_scaled[:, i]
    
    print("✅ RobustScaler applied (median + IQR)")
    return data_scaled, scaler

# =========================================================================
# STEP 3: CORRELATION ANALYSIS
# =========================================================================
def correlation_analysis(data_scaled, features, target):
    """Spearman correlation analysis"""
    print("\n📈 STEP 3: SPEARMAN CORRELATION")
    print("-" * 40)
    
    correlations = {}
    scaled_features = [f'{f}_scaled' for f in features]
    
    for orig_feat, scaled_feat in zip(features, scaled_features):
        x = data_scaled[scaled_feat].values
        y = data_scaled[target].values
        
        # Remove NaN and calculate correlation
        mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(mask) > 1:
            rho, p_value = stats.spearmanr(x[mask], y[mask])
            correlations[scaled_feat] = {
                'original': orig_feat,
                'correlation': rho,
                'abs_correlation': abs(rho)
            }
            print(f"   {orig_feat:25}: ρ = {rho:7.4f}")
    
    # Select top 6 features
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
    top_6_scaled = [item[0] for item in sorted_corrs[:6]]
    top_6_original = [item[1]['original'] for item in sorted_corrs[:6]]
    
    print(f"\n🎯 TOP 6 FEATURES SELECTED:")
    for i, feat in enumerate(top_6_original, 1):
        corr = correlations[f'{feat}_scaled']['abs_correlation']
        print(f"   {i}. {feat:25}: |ρ| = {corr:.4f}")
    
    return top_6_scaled, top_6_original

# =========================================================================
# STEP 4: PCA ANALYSIS AND SELECTION
# =========================================================================
def pca_analysis(data_scaled, top_6_scaled, top_6_original):
    """Apply PCA and determine optimal number of components"""
    print("\n🔄 STEP 4: PCA ANALYSIS AND SELECTION")
    print("-" * 50)
    
    # Prepare data for PCA
    X_for_pca = data_scaled[top_6_scaled].values
    
    print(f"📊 Input for PCA:")
    print(f"   ✅ Features: {len(top_6_scaled)} (top 6 by correlation)")
    print(f"   ✅ Data shape: {X_for_pca.shape}")
    print(f"   ✅ Selected features: {', '.join(top_6_original)}")
    
    # Apply PCA with all components first to see explained variance
    pca_full = PCA()
    pca_full.fit(X_for_pca)
    
    # Display explained variance for each component
    print(f"\n📈 PCA Explained Variance Analysis:")
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    for i, (var_ratio, cum_var) in enumerate(zip(pca_full.explained_variance_ratio_, cumulative_variance)):
        print(f"   PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    # Determine optimal number of components
    # Method 1: 80% variance threshold
    components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    
    # Method 2: 90% variance threshold  
    components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    
    # Method 3: Elbow method (find where variance gain becomes smaller)
    variance_diff = np.diff(pca_full.explained_variance_ratio_)
    elbow_point = np.argmax(variance_diff < 0.05) + 2  # +2 because of diff and 1-indexing
    if elbow_point == 2:  # If no clear elbow found, use a reasonable default
        elbow_point = min(4, len(top_6_scaled))
    
    print(f"\n🎯 PCA Component Selection:")
    print(f"   📊 80% variance: {components_80} components ({cumulative_variance[components_80-1]*100:.2f}%)")
    print(f"   📊 90% variance: {components_90} components ({cumulative_variance[components_90-1]*100:.2f}%)")
    print(f"   📐 Elbow method: {elbow_point} components")
    
    # Choose optimal number of components (prefer 80% threshold, but at least 2)
    optimal_components = max(2, min(components_80, 4))  # Between 2-4 components
    
    print(f"   🏆 Selected: {optimal_components} components ({cumulative_variance[optimal_components-1]*100:.2f}% variance)")
    
    # Apply PCA with optimal number of components
    pca_optimal = PCA(n_components=optimal_components)
    X_pca = pca_optimal.fit_transform(X_for_pca)
    
    # Create PCA component names
    pca_features = [f'PC{i+1}' for i in range(optimal_components)]
    
    print(f"\n✅ PCA Transformation Complete:")
    print(f"   📐 Reduced from {len(top_6_scaled)} to {optimal_components} dimensions")
    print(f"   📊 Final explained variance: {cumulative_variance[optimal_components-1]*100:.2f}%")
    print(f"   🔧 PCA features: {', '.join(pca_features)}")
    
    # Feature loadings analysis
    print(f"\n📋 PCA Component Loadings (Top contributors):")
    loadings = pca_optimal.components_
    
    for i in range(optimal_components):
        print(f"\n   PC{i+1} (explains {pca_optimal.explained_variance_ratio_[i]*100:.2f}% variance):")
        
        # Get feature contributions for this component
        component_loadings = [(top_6_original[j], abs(loadings[i, j])) for j in range(len(top_6_original))]
        component_loadings.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 3 contributors
        for j, (feature, loading) in enumerate(component_loadings[:3]):
            print(f"     {j+1}. {feature:25}: {loading:.4f}")
    
    # Visualization
    if optimal_components >= 2:
        fig = plt.figure(figsize=(15, 10))
        
        # Subplot 1: Explained variance
        plt.subplot(2, 3, 1)
        plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                pca_full.explained_variance_ratio_, alpha=0.7, color='skyblue')
        plt.axhline(y=0.8/len(pca_full.explained_variance_ratio_), color='red', linestyle='--', alpha=0.7)
        plt.title('PCA Explained Variance by Component', fontweight='bold')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Cumulative explained variance
        plt.subplot(2, 3, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        plt.axvline(x=optimal_components, color='green', linestyle='-', linewidth=2, label=f'Selected: {optimal_components}')
        plt.title('Cumulative Explained Variance', fontweight='bold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: PCA scatter plot (PC1 vs PC2)
        plt.subplot(2, 3, 3)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
        plt.title(f'PCA Space: PC1 vs PC2\n({pca_optimal.explained_variance_ratio_[0]*100:.1f}% vs {pca_optimal.explained_variance_ratio_[1]*100:.1f}% variance)', 
                 fontweight='bold')
        plt.xlabel(f'PC1 ({pca_optimal.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca_optimal.explained_variance_ratio_[1]*100:.1f}%)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Feature loadings heatmap
        plt.subplot(2, 3, 4)
        loadings_df = pd.DataFrame(loadings.T, 
                                 columns=[f'PC{i+1}' for i in range(optimal_components)],
                                 index=[feat.replace('_scaled', '') for feat in top_6_scaled])
        
        im = plt.imshow(loadings_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, shrink=0.8)
        plt.title('PCA Loadings Heatmap', fontweight='bold')
        plt.xlabel('Principal Components')
        plt.ylabel('Original Features')
        plt.xticks(range(optimal_components), [f'PC{i+1}' for i in range(optimal_components)])
        plt.yticks(range(len(loadings_df.index)), [feat[:15] + '...' if len(feat) > 15 else feat for feat in loadings_df.index])
        
        # Add text annotations for strong loadings
        for i in range(len(loadings_df.index)):
            for j in range(optimal_components):
                if abs(loadings_df.values[i, j]) > 0.5:
                    plt.text(j, i, f'{loadings_df.values[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold', color='white')
        
        # Subplot 5: Component contributions pie chart for PC1
        plt.subplot(2, 3, 5)
        pc1_contributions = [(top_6_original[i], abs(loadings[0, i])) for i in range(len(top_6_original))]
        pc1_contributions.sort(key=lambda x: x[1], reverse=True)
        
        labels = [item[0][:10] + '...' if len(item[0]) > 10 else item[0] for item in pc1_contributions[:4]]
        sizes = [item[1] for item in pc1_contributions[:4]]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'PC1 Feature Contributions\n({pca_optimal.explained_variance_ratio_[0]*100:.1f}% of total variance)', 
                 fontweight='bold')
        
        # Subplot 6: Component contributions pie chart for PC2
        plt.subplot(2, 3, 6)
        pc2_contributions = [(top_6_original[i], abs(loadings[1, i])) for i in range(len(top_6_original))]
        pc2_contributions.sort(key=lambda x: x[1], reverse=True)
        
        labels = [item[0][:10] + '...' if len(item[0]) > 10 else item[0] for item in pc2_contributions[:4]]
        sizes = [item[1] for item in pc2_contributions[:4]]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'PC2 Feature Contributions\n({pca_optimal.explained_variance_ratio_[1]*100:.1f}% of total variance)', 
                 fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 PCA VISUALIZATION SUMMARY:")
        print(f"   📊 Explained Variance: Shows individual and cumulative variance")
        print(f"   🎯 PCA Space: Data distribution in PC1-PC2 plane")
        print(f"   🔥 Loadings Heatmap: Feature contributions to each PC")
        print(f"   📋 Pie Charts: Top contributors to PC1 and PC2")
    
    return X_pca, pca_optimal, pca_features, optimal_components

# =========================================================================
# STEP 5: CLUSTERING ANALYSIS
# =========================================================================
def clustering_analysis(X_clustering, pca_features):
    """Elbow method and cluster visualization in PCA space"""
    print("\n📐 STEP 5: CLUSTERING ANALYSIS (IN PCA SPACE)")
    print("-" * 50)
    
    # Test different K values
    k_range = range(2, 11)
    sse_values = []
    silhouette_scores = []
    kmeans_models = {}
    
    # Variables untuk tracking K pertama dengan silhouette ≥ 0.5 (minimal K=3)
    first_k_above_05 = None
    silhouette_threshold = 0.5
    min_k_threshold = 3  # Minimum K value to consider
    
    print("Testing K values in PCA space:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_clustering)
        
        sse = kmeans.inertia_
        sil_score = silhouette_score(X_clustering, labels)
        
        sse_values.append(sse)
        silhouette_scores.append(sil_score)
        kmeans_models[k] = {'model': kmeans, 'labels': labels}
        
        # Check jika ini adalah K pertama yang mencapai threshold DAN >= minimal K
        if first_k_above_05 is None and sil_score >= silhouette_threshold and k >= min_k_threshold:
            first_k_above_05 = k
        
        print(f"   K={k}: SSE={sse:.1f}, Silhouette={sil_score:.4f}")
    
    # Find optimal K
    max_sil_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[max_sil_idx]
    
    print(f"\n🎯 Optimal K = {optimal_k} (best silhouette: {max(silhouette_scores):.4f})")
    
    # Display K pertama yang mencapai silhouette ≥ 0.5 dengan minimal K=3
    if first_k_above_05 is not None:
        first_k_silhouette = silhouette_scores[list(k_range).index(first_k_above_05)]
        print(f"🔥 FIRST K ≥ 0.5 SILHOUETTE (MIN K=3): K = {first_k_above_05} (silhouette = {first_k_silhouette:.4f})")
        print(f"   📍 This is the first K value ≥ {min_k_threshold} that reaches or exceeds silhouette score of {silhouette_threshold}")
    else:
        print(f"⚠️  No K value ≥ {min_k_threshold} reached silhouette score ≥ {silhouette_threshold}")
        print(f"   📊 Maximum silhouette achieved: {max(silhouette_scores):.4f} at K={optimal_k}")
    
    # Enhanced Visualization in PCA space
    if first_k_above_05 is not None and X_clustering.shape[1] >= 2:
        # Create comprehensive visualization
        fig = plt.figure(figsize=(15, 6))
        
        # Subplot 1: Elbow plot
        plt.subplot(1, 2, 1)
        plt.plot(k_range, sse_values, 'bo-', linewidth=2, markersize=8)
        
        # Highlight K pertama ≥ 0.5 di elbow plot
        if first_k_above_05 in k_range:
            k_idx = list(k_range).index(first_k_above_05)
            plt.scatter([first_k_above_05], [sse_values[k_idx]], 
                       color='orange', s=150, marker='*', zorder=5, 
                       label=f'First K≥0.5: {first_k_above_05}')
        
        plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Cluster visualization untuk K pertama ≥ 0.5
        plt.subplot(1, 2, 2)
        
        cluster_info = kmeans_models[first_k_above_05]
        labels = cluster_info['labels']
        centroids = cluster_info['model'].cluster_centers_
        
        # Colors untuk clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        # Plot clusters in PCA space
        for cluster_id in range(first_k_above_05):
            mask = labels == cluster_id
            plt.scatter(X_clustering[mask, 0], X_clustering[mask, 1], 
                       c=colors[cluster_id % len(colors)], alpha=0.7, s=50, 
                       label=f'Cluster {cluster_id}')
        
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='black', marker='X', s=300, linewidths=2, 
                   label='Centroids', edgecolors='white')
        
        # Get silhouette score untuk title
        k_idx = list(k_range).index(first_k_above_05)
        sil_score = silhouette_scores[k_idx]
        
        plt.title(f'PCA Clustering: K={first_k_above_05}\n(Silhouette = {sil_score:.4f})', 
                 fontsize=14, fontweight='bold', color='darkorange')
        plt.xlabel(f'{pca_features[0]} ({X_clustering.shape[1]}D PCA)', fontsize=12)
        plt.ylabel(f'{pca_features[1]}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 VISUALIZATION SUMMARY:")
        print(f"   🔍 Elbow Method: Shows SSE reduction across K values")
        print(f"   ⭐ Highlighted: K={first_k_above_05} (First to reach silhouette ≥ 0.5)")
        print(f"   📊 Cluster Plot: Shows {first_k_above_05} clusters in PCA space")
        print(f"   🎯 Silhouette Score: {silhouette_scores[list(k_range).index(first_k_above_05)]:.4f}")
    
    elif first_k_above_05 is None:
        # Jika tidak ada K yang mencapai 0.5, tampilkan basic elbow plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_range, sse_values, 'bo-', linewidth=2, markersize=8)
        plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.5 threshold')
        plt.title('Silhouette Scores by K', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 BASIC VISUALIZATION (No K ≥ 0.5 found):")
        print(f"   🔍 Elbow Method: Shows SSE reduction across K values")
        print(f"   ⚠️  No K value ≥ {min_k_threshold} reached silhouette ≥ 0.5")
        print(f"   📊 Best silhouette: {max(silhouette_scores):.4f} at K={optimal_k}")
    
    # Detailed analysis for K=4,5,6 dan K pertama ≥ 0.5
    analysis_k_values = [4, 5, 6]
    if first_k_above_05 is not None and first_k_above_05 not in analysis_k_values:
        analysis_k_values.append(first_k_above_05)
    
    print(f"\n📊 DETAILED PCA CLUSTERING ANALYSIS:")
    for k in sorted(analysis_k_values):
        if k in kmeans_models:
            labels = kmeans_models[k]['labels']
            k_idx = list(k_range).index(k)
            sil_score = silhouette_scores[k_idx]
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Special marker untuk K pertama ≥ 0.5 dengan min K=3
            marker = ""
            if k == first_k_above_05:
                marker = " 🔥 (FIRST K ≥ 0.5 in PCA space)"
            elif k in [4, 5, 6]:
                marker = " (ANALYSIS)"
            
            print(f"\n   K={k} in PCA space (Silhouette: {sil_score:.4f}){marker}:")
            for cluster_id, count in zip(unique_labels, counts):
                percentage = count / len(labels) * 100
                print(f"     Cluster {cluster_id}: {count:3d} points ({percentage:5.1f}%)")
    
    return optimal_k, kmeans_models, silhouette_scores, first_k_above_05

# =========================================================================
# STEP 6: PREPARE DATA FOR SUPERVISED LEARNING
# =========================================================================
def prepare_supervised_features(data_clean, data_scaled, X_pca, first_k_above_05, kmeans_models, scaler, pca_optimal, features, target):
    """Prepare enhanced dataset with cluster features for supervised learning"""
    print("\n🔗 STEP 6: PREPARE FEATURES FOR SUPERVISED LEARNING")
    print("-" * 60)
    
    # Create enhanced dataset with original features + cluster features
    enhanced_data = data_clean.copy()
    
    # Add cluster ID as new feature if clustering was successful
    if first_k_above_05 is not None and first_k_above_05 in kmeans_models:
        cluster_labels = kmeans_models[first_k_above_05]['labels']
        enhanced_data['Cluster_ID'] = cluster_labels
        
        print(f"✅ Added Cluster_ID feature (K={first_k_above_05})")
        print("   📊 Cluster distribution:")
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique_clusters, counts):
            percentage = count / len(cluster_labels) * 100
            print(f"     Cluster {cluster_id}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Add PCA components as features
    if X_pca is not None:
        pca_components = X_pca.shape[1]
        for i in range(pca_components):
            enhanced_data[f'PC{i+1}'] = X_pca[:, i]
        
        print(f"✅ Added {pca_components} PCA component features")
    
    # Create cluster-based features (if clustering available)
    if first_k_above_05 is not None and first_k_above_05 in kmeans_models:
        cluster_labels = kmeans_models[first_k_above_05]['labels']
        
        # Distance to cluster centroid
        centroids = kmeans_models[first_k_above_05]['model'].cluster_centers_
        cluster_distances = []
        
        for i, label in enumerate(cluster_labels):
            point = X_pca[i]
            centroid = centroids[label]
            distance = np.linalg.norm(point - centroid)
            cluster_distances.append(distance)
        
        enhanced_data['Distance_to_Centroid'] = cluster_distances
        print(f"✅ Added Distance_to_Centroid feature")
        
        # Cluster-based statistics
        cluster_delivery_means = {}
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_delivery_mean = enhanced_data.loc[cluster_mask, target].mean()
            cluster_delivery_means[cluster_id] = cluster_delivery_mean
        
        # Add cluster's historical average delivery time as feature
        cluster_avg_delivery = [cluster_delivery_means[label] for label in cluster_labels]
        enhanced_data['Cluster_Avg_Delivery'] = cluster_avg_delivery
        print(f"✅ Added Cluster_Avg_Delivery feature")
    
    # Save models for supervised learning use
    models_dict = {
        'scaler': scaler,
        'pca_model': pca_optimal,
        'kmeans_model': kmeans_models[first_k_above_05]['model'] if first_k_above_05 in kmeans_models else None,
        'optimal_k': first_k_above_05,
        'feature_names': {
            'original': [col for col in data_clean.columns if col != target],
            'enhanced': [col for col in enhanced_data.columns if col != target]
        }
    }
    
    print(f"\n📋 ENHANCED DATASET SUMMARY:")
    print(f"   📊 Original features: {len([col for col in data_clean.columns if col != target])}")
    print(f"   📊 Enhanced features: {len([col for col in enhanced_data.columns if col != target])}")
    print(f"   ➕ Added features: {len(enhanced_data.columns) - len(data_clean.columns)}")
    print(f"   🎯 Target variable: {target}")
    
    return enhanced_data, models_dict

# =========================================================================
# STEP 7: SAVE RESULTS AND EXPORT FOR SUPERVISED
# =========================================================================
def save_results_and_export(data_clean, data_scaled, enhanced_data, models_dict, top_6_original, 
                           optimal_k, first_k_above_05, kmeans_models, X_pca, pca_optimal, pca_features, features, target):
    """Save results and export data for supervised learning"""
    print("\n💾 STEP 7: SAVE RESULTS AND EXPORT FOR SUPERVISED")
    print("-" * 60)
    
    filename = 'Pizza_Analysis_PCA_Enhanced_Results.xlsx'
    supervised_filename = 'Pizza_Enhanced_Data_for_Supervised.xlsx'
    
    # Save unsupervised analysis results (original format)
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Original data
        data_clean.to_excel(writer, sheet_name='Original_Data', index=False)
        
        # Scaled data
        data_scaled.to_excel(writer, sheet_name='Scaled_Data', index=False)
        
        # PCA information
        pca_info = {
            'Component': [f'PC{i+1}' for i in range(len(pca_optimal.explained_variance_ratio_))],
            'Explained_Variance_Ratio': pca_optimal.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca_optimal.explained_variance_ratio_),
            'Explained_Variance': pca_optimal.explained_variance_
        }
        pca_df = pd.DataFrame(pca_info)
        pca_df.to_excel(writer, sheet_name='PCA_Information', index=False)
        
        # PCA loadings
        loadings_data = []
        for i, feature in enumerate(top_6_original):
            row = {'Original_Feature': feature}
            for j, pc in enumerate(pca_features):
                row[pc] = pca_optimal.components_[j, i]
            loadings_data.append(row)
        
        loadings_df = pd.DataFrame(loadings_data)
        loadings_df.to_excel(writer, sheet_name='PCA_Loadings', index=False)
        
        # PCA transformed data
        pca_data = pd.DataFrame(X_pca, columns=pca_features)
        pca_data.to_excel(writer, sheet_name='PCA_Transformed', index=False)
        
        # Clustering results untuk K pertama ≥ 0.5
        if first_k_above_05 is not None and first_k_above_05 in kmeans_models:
            print(f"📊 Saving PCA clustering results for K={first_k_above_05} (First K ≥ 0.5 Silhouette)")
            
            # Get cluster labels untuk K pertama ≥ 0.5
            cluster_labels = kmeans_models[first_k_above_05]['labels']
            
            # Create DataFrame dengan data asli + cluster assignment + PCA coordinates
            clustering_results = data_clean.copy()
            clustering_results['Cluster_ID'] = cluster_labels
            clustering_results['K_Value'] = first_k_above_05
            clustering_results['Clustering_Method'] = f'PCA_First_K_Above_05_Silhouette'
            
            # Tambahkan PCA coordinates
            for i, pc_name in enumerate(pca_features):
                clustering_results[pc_name] = X_pca[:, i]
            
            # Save clustering results
            clustering_results.to_excel(writer, sheet_name=f'PCA_Clustering_K{first_k_above_05}', index=False)
        
        # Enhanced summary dengan PCA information
        summary_data = [
            {'Step': 'Data Preparation', 'Result': f'{len(data_clean)} rows, 14 features (8 original + 6 interaction)'},
            {'Step': 'Feature Engineering', 'Result': 'Distance-Traffic + Pizza Profile interaction features added'},
            {'Step': 'Feature Selection', 'Result': f'Top 6 features: {", ".join(top_6_original)}'},
            {'Step': 'PCA Analysis', 'Result': f'{len(pca_features)} components, {pca_optimal.explained_variance_ratio_.sum()*100:.1f}% variance'},
            {'Step': 'Clustering Method', 'Result': f'K-Means clustering in {len(pca_features)}D PCA space'},
            {'Step': 'PCA Features', 'Result': f'{", ".join(pca_features)}'},
            {'Step': 'Optimal K', 'Result': f'K = {optimal_k}'},
            {'Step': 'First K ≥ 0.5 Silhouette (min K=3)', 'Result': f'K = {first_k_above_05}' if first_k_above_05 else 'None found'},
            {'Step': 'Enhanced Features Created', 'Result': f'{len(enhanced_data.columns) - len(data_clean.columns)} new features'},
            {'Step': 'Ready for Supervised', 'Result': f'Enhanced dataset with {len(enhanced_data.columns)} total features'}
        ]
        summary = pd.DataFrame(summary_data)
        summary.to_excel(writer, sheet_name='PCA_Summary', index=False)
    
    # Export enhanced data for supervised learning
    with pd.ExcelWriter(supervised_filename, engine='openpyxl') as writer:
        # Enhanced dataset
        enhanced_data.to_excel(writer, sheet_name='Enhanced_Data', index=False)
        
        # Feature documentation
        feature_docs = []
        
        # Original features
        for feature in data_clean.columns:
            if feature != target:
                feature_docs.append({
                    'Feature_Name': feature,
                    'Feature_Type': 'Original',
                    'Description': f'Original feature from dataset',
                    'Source': 'Raw data'
                })
        
        # Target variable
        feature_docs.append({
            'Feature_Name': target,
            'Feature_Type': 'Target',
            'Description': 'Target variable for supervised learning',
            'Source': 'Raw data'
        })
        
        # Cluster features
        if first_k_above_05 is not None:
            feature_docs.append({
                'Feature_Name': 'Cluster_ID',
                'Feature_Type': 'Cluster',
                'Description': f'Cluster assignment from K-Means (K={first_k_above_05})',
                'Source': 'Unsupervised clustering'
            })
            
            feature_docs.append({
                'Feature_Name': 'Distance_to_Centroid',
                'Feature_Type': 'Cluster',
                'Description': 'Euclidean distance to cluster centroid in PCA space',
                'Source': 'Unsupervised clustering'
            })
            
            feature_docs.append({
                'Feature_Name': 'Cluster_Avg_Delivery',
                'Feature_Type': 'Cluster',
                'Description': 'Historical average delivery time for the cluster',
                'Source': 'Unsupervised clustering'
            })
        
        # PCA features
        if 'PC1' in enhanced_data.columns:
            pca_components = len([col for col in enhanced_data.columns if col.startswith('PC')])
            for i in range(pca_components):
                feature_docs.append({
                    'Feature_Name': f'PC{i+1}',
                    'Feature_Type': 'PCA',
                    'Description': f'Principal Component {i+1} from PCA transformation',
                    'Source': 'PCA dimensionality reduction'
                })
        
        feature_docs_df = pd.DataFrame(feature_docs)
        feature_docs_df.to_excel(writer, sheet_name='Feature_Documentation', index=False)
        
        # Models info for supervised use
        models_info = [
            {'Component': 'Scaler', 'Type': 'RobustScaler', 'Purpose': 'Feature scaling for new data'},
            {'Component': 'PCA', 'Type': 'PCA', 'Purpose': 'Dimensionality reduction transformation'},
            {'Component': 'KMeans', 'Type': f'KMeans(n_clusters={first_k_above_05})', 'Purpose': 'Cluster assignment for new data'},
            {'Component': 'Optimal_K', 'Type': 'Integer', 'Purpose': f'Selected K value: {first_k_above_05}'}
        ]
        models_info_df = pd.DataFrame(models_info)
        models_info_df.to_excel(writer, sheet_name='Models_Info', index=False)
    
    print(f"✅ Unsupervised analysis results saved to '{filename}'")
    print(f"✅ Enhanced dataset for supervised learning saved to '{supervised_filename}'")
    print(f"\n🔗 CONNECTION TO SUPERVISED LEARNING:")
    print(f"   📁 Use '{supervised_filename}' as input for supervised learning")
    print(f"   📊 Enhanced features include: Cluster_ID, PC components, Distance_to_Centroid")
    print(f"   🎯 Target variable: {target}")
    print(f"   🔧 Models saved for prediction pipeline: scaler, pca, kmeans")
    
    return models_dict

# =========================================================================
# MAIN EXECUTION WITH PCA + SUPERVISED CONNECTION
# =========================================================================
def main():
    """Main execution function with PCA integration + supervised connection"""
    try:
        print("🎯 PIZZA DELIVERY UNSUPERVISED ANALYSIS")
        print("🔗 Creating enhanced features for supervised learning")
        print("="*80)
        
        # Step 1: Load and prepare data (with feature engineering)
        data_clean, features, target = load_and_prepare_data()
        
        # Step 2: Apply scaling
        data_scaled, scaler = apply_scaling(data_clean, features)
        
        # Step 3: Correlation analysis
        top_6_scaled, top_6_original = correlation_analysis(data_scaled, features, target)
        
        # Step 4: PCA analysis
        X_pca, pca_optimal, pca_features, optimal_components = pca_analysis(data_scaled, top_6_scaled, top_6_original)
        
        # Step 5: Clustering analysis (in PCA space)
        optimal_k, kmeans_models, silhouette_scores, first_k_above_05 = clustering_analysis(X_pca, pca_features)
        
        # Step 6: Prepare features for supervised learning
        enhanced_data, models_dict = prepare_supervised_features(data_clean, data_scaled, X_pca, 
                                                               first_k_above_05, kmeans_models, scaler, pca_optimal, features, target)
        
        # Step 7: Save results and export for supervised
        final_models_dict = save_results_and_export(data_clean, data_scaled, enhanced_data, models_dict, 
                                                   top_6_original, optimal_k, first_k_above_05, kmeans_models, 
                                                   X_pca, pca_optimal, pca_features, features, target)
        
        # Final summary
        print(f"\n" + "="*80)
        print("🎉 UNSUPERVISED ANALYSIS COMPLETED (WITH SUPERVISED CONNECTION)!")
        print("="*80)
        print(f"📊 Pipeline: Feature Engineering → RobustScaler → Spearman → PCA → K-Means Clustering")
        print(f"🔧 Enhanced: Added 6 interaction features + PCA dimensionality reduction")
        print(f"✅ Features: {len(features)} total → Top 6 → {len(pca_features)} PCA components → Clustering")
        print(f"📐 PCA: {optimal_components} components explaining {pca_optimal.explained_variance_ratio_.sum()*100:.1f}% variance")
        print(f"🎯 Components: {', '.join(pca_features)}")
        print(f"📊 Optimal Clusters: K = {optimal_k}")
        if first_k_above_05 is not None:
            print(f"🔥 First K ≥ 0.5 Silhouette (min K=3): K = {first_k_above_05}")
            print(f"✅ Enhanced features: {len(enhanced_data.columns)} total features")
        else:
            print(f"⚠️  No K ≥ 3 reached silhouette ≥ 0.5")
        print(f"💾 Results saved to: Pizza_Analysis_PCA_Enhanced_Results.xlsx")
        print(f"🔗 Enhanced data for supervised: Pizza_Enhanced_Data_for_Supervised.xlsx")
        
        print(f"\n🚀 NEXT STEPS FOR SUPERVISED LEARNING:")
        print(f"   1. Load 'Pizza_Enhanced_Data_for_Supervised.xlsx'")
        print(f"   2. Use enhanced features including Cluster_ID as input")
        print(f"   3. Train regression model to predict '{target}'")
        print(f"   4. Leverage cluster information for better predictions")
        print("="*80)
        
        return enhanced_data, final_models_dict
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your data file 'Train Data.xlsx' exists and has the correct format.")
        import traceback
        traceback.print_exc()

# Run the analysis
if __name__ == "__main__":
    enhanced_data, models_dict = main() 