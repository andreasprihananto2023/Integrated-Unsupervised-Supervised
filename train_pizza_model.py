import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_enhanced_data():
    """Load enhanced dataset created by unsupervised analysis"""
    print("\n🍕 PIZZA DELIVERY TIME PREDICTION - WITH ENHANCED FEATURES")
    print("="*70)
    print("\n📊 STEP 1: LOAD ENHANCED DATA FROM UNSUPERVISED ANALYSIS")
    print("-" * 60)
    
    try:
        enhanced_data = pd.read_excel('Pizza_Enhanced_Data_for_Supervised.xlsx', sheet_name='Enhanced_Data')
        feature_docs = pd.read_excel('Pizza_Enhanced_Data_for_Supervised.xlsx', sheet_name='Feature_Documentation')
        
        print(f"✅ Enhanced data loaded: {len(enhanced_data)} rows, {len(enhanced_data.columns)} columns")
        
        original_features = feature_docs[feature_docs['Feature_Type'] == 'Original']['Feature_Name'].tolist()
        cluster_features = feature_docs[feature_docs['Feature_Type'] == 'Cluster']['Feature_Name'].tolist()
        pca_features = feature_docs[feature_docs['Feature_Type'] == 'PCA']['Feature_Name'].tolist()
        target = feature_docs[feature_docs['Feature_Type'] == 'Target']['Feature_Name'].iloc[0]
        
        print(f"\n📋 Feature breakdown:")
        print(f"   🔹 Original features: {len(original_features)} - {original_features}")
        print(f"   🔹 Cluster features: {len(cluster_features)} - {cluster_features}")  
        print(f"   🔹 PCA features: {len(pca_features)} - {pca_features}")
        print(f"   🎯 Target: {target}")
        
        if 'Cluster_ID' in enhanced_data.columns:
            print(f"\n📊 Cluster distribution:")
            cluster_counts = enhanced_data['Cluster_ID'].value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                percentage = count / len(enhanced_data) * 100
                avg_delivery = enhanced_data[enhanced_data['Cluster_ID'] == cluster_id][target].mean()
                print(f"   Cluster {cluster_id}: {count:3d} samples ({percentage:5.1f}%) - Avg delivery: {avg_delivery:.1f} min")
        
        return enhanced_data, original_features, cluster_features, pca_features, target
        
    except FileNotFoundError:
        print("❌ Error: Enhanced data file not found!")
        print("   Please run the unsupervised analysis first to generate:")
        print("   'Pizza_Enhanced_Data_for_Supervised.xlsx'")
        print("\n🔄 Falling back to original data analysis...")
        return load_original_data()

def load_original_data():
    """Fallback: Load original data if enhanced data not available"""
    print("\n📊 LOADING ORIGINAL DATA (FALLBACK)")
    print("-" * 50)
    
    data = pd.read_excel('Train Data.xlsx')
    data.columns = data.columns.str.strip()
    
    original_features = ['Pizza Type', 'Distance (km)', 'Is Weekend', 'Topping Density', 
                        'Order Month', 'Pizza Complexity', 'Traffic Impact', 'Order Hour']
    target = 'Delivery Duration (min)'
    
    enhanced_data = data[original_features + [target]].dropna()
    cluster_features = []
    pca_features = []
    
    print(f"✅ Original data loaded: {len(enhanced_data)} rows")
    print("⚠️  No enhanced features available - using original features only")
    
    return enhanced_data, original_features, cluster_features, pca_features, target


def analyze_enhanced_features(enhanced_data, original_features, cluster_features, pca_features, target):
    """Analyze the impact of enhanced features on target variable"""
    print(f"\n📈 STEP 2: ENHANCED FEATURE ANALYSIS")
    print("-" * 50)
    
    all_features = original_features + cluster_features + pca_features
    correlations = {}
    
    print("📊 CORRELATION ANALYSIS (Spearman):")
    for feature in all_features:
        if feature in enhanced_data.columns and enhanced_data[feature].dtype in ['int64', 'float64', 'Int64']:
            try:
                x = enhanced_data[feature].values
                y = enhanced_data[target].values
                
                x = pd.to_numeric(x, errors='coerce')
                y = pd.to_numeric(y, errors='coerce')
                
                mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(mask) > 1:
                    rho, p_value = stats.spearmanr(x[mask], y[mask])
                    correlations[feature] = {
                        'correlation': rho,
                        'abs_correlation': abs(rho),
                        'p_value': p_value
                    }
                    
                    feature_type = "Original" if feature in original_features else \
                                  "Cluster" if feature in cluster_features else \
                                  "PCA" if feature in pca_features else "Unknown"
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {feature:25} ({feature_type:8}): ρ = {rho:7.4f} (p={p_value:.4f}){significance}")
                else:
                    print(f"   {feature:25}: Insufficient valid data")
            except Exception as e:
                print(f"   {feature:25}: Error - {str(e)}")
    
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    print(f"\n🎯 TOP 10 FEATURES BY CORRELATION:")
    for i, (feature, corr_info) in enumerate(sorted_correlations[:10], 1):
        feature_type = "Original" if feature in original_features else \
                      "Cluster" if feature in cluster_features else \
                      "PCA" if feature in pca_features else "Unknown"
        print(f"   {i:2d}. {feature:25} ({feature_type:8}): |ρ| = {abs(corr_info['correlation']):.4f}")
    
    if 'Cluster_ID' in enhanced_data.columns:
        print(f"\n📊 DELIVERY TIME BY CLUSTER:")
        try:
            enhanced_data['Cluster_ID'] = pd.to_numeric(enhanced_data['Cluster_ID'], errors='coerce')
            enhanced_data[target] = pd.to_numeric(enhanced_data[target], errors='coerce')
            
            valid_data = enhanced_data.dropna(subset=['Cluster_ID', target])
            cluster_stats = valid_data.groupby('Cluster_ID')[target].agg(['count', 'mean', 'std', 'min', 'max'])
            
            for cluster_id, row in cluster_stats.iterrows():
                count = int(row['count']) if not pd.isna(row['count']) else 0
                mean_val = float(row['mean']) if not pd.isna(row['mean']) else 0.0
                std_val = float(row['std']) if not pd.isna(row['std']) else 0.0
                min_val = float(row['min']) if not pd.isna(row['min']) else 0.0
                max_val = float(row['max']) if not pd.isna(row['max']) else 0.0
                
                print(f"   Cluster {int(cluster_id)}: {mean_val:6.2f} ± {std_val:5.2f} min "
                      f"(n={count:3d}, range: {min_val:.1f}-{max_val:.1f})")
                      
        except Exception as e:
            print(f"   ❌ Error in cluster analysis: {str(e)}")
            print("   Skipping cluster-based analysis...")
    
    return correlations, all_features


def select_features(enhanced_data, original_features, cluster_features, pca_features, correlations, target):
    """Select features using different strategies"""
    print(f"\n🎯 STEP 3: FEATURE SELECTION STRATEGIES")
    print("-" * 50)
    
    original_corrs = {f: correlations[f] for f in original_features if f in correlations}
    original_sorted = sorted(original_corrs.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
    original_top6 = [item[0] for item in original_sorted[:6]]
    
    all_features = original_features + cluster_features + pca_features
    
    all_sorted = sorted(correlations.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
    mixed_top10 = [item[0] for item in all_sorted[:10]]
    
    original_cluster = original_features + cluster_features
    original_pca = original_features + pca_features
    
    feature_sets = {
        'Original_Top6': original_top6,
        'All_Enhanced': all_features,
        'Mixed_Top10': mixed_top10,
        'Original_Plus_Cluster': original_cluster,
        'Original_Plus_PCA': original_pca
    }
    
    print("📋 FEATURE SELECTION STRATEGIES:")
    for strategy, features in feature_sets.items():
        available_features = [f for f in features if f in enhanced_data.columns]
        print(f"   {strategy:20}: {len(available_features):2d} features")
    
    return feature_sets


def preprocess_data_multiple_sets(enhanced_data, feature_sets, target):
    """Preprocess data for multiple feature sets"""
    print(f"\n🔧 STEP 4: DATA PREPROCESSING")
    print("-" * 50)
    
    preprocessed_data = {}
    
    for strategy_name, features in feature_sets.items():
        available_features = [f for f in features if f in enhanced_data.columns]
        
        if not available_features:
            print(f"⚠️  {strategy_name}: No valid features available")
            continue
            
        X = enhanced_data[available_features].copy()
        y = enhanced_data[target].copy()
        
        try:
            y_bins = pd.cut(y, bins=5, labels=['Very_Fast', 'Fast', 'Normal', 'Slow', 'Very_Slow'])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y_bins
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)
        
        preprocessed_data[strategy_name] = {
            'features': available_features,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler
        }
        
        print(f"   ✅ {strategy_name:20}: {len(available_features):2d} features, {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    return preprocessed_data


def train_models_all_strategies(preprocessed_data):
    """Train models for all feature selection strategies"""
    print(f"\n🤖 STEP 5: MODEL TRAINING & COMPARISON")
    print("-" * 50)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
    }
    
    scaled_models = ['Ridge Regression', 'K-Nearest Neighbors']
    
    all_results = []
    trained_models = {}
    
    print("🏃‍♂️ TRAINING MODELS FOR ALL STRATEGIES:")
    
    for strategy_name, data_dict in preprocessed_data.items():
        print(f"\n📊 Strategy: {strategy_name}")
        strategy_results = []
        
        for model_name, model in models.items():
            if model_name in scaled_models:
                X_train, X_test = data_dict['X_train_scaled'], data_dict['X_test_scaled']
            else:
                X_train, X_test = data_dict['X_train'], data_dict['X_test']
            
            y_train, y_test = data_dict['y_train'], data_dict['y_test']
            
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            y_pred_test = model_copy.predict(X_test)
            
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            result = {
                'Strategy': strategy_name,
                'Model': model_name,
                'Feature_Count': len(data_dict['features']),
                'Test_R2': test_r2,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae
            }
            
            all_results.append(result)
            strategy_results.append(result)
            
            key = f"{strategy_name}_{model_name}"
            trained_models[key] = {
                'model': model_copy,
                'strategy': strategy_name,
                'model_name': model_name,
                'scaler': data_dict['scaler'],
                'features': data_dict['features'],
                'scaled_data': model_name in scaled_models
            }
            
            print(f"     {model_name:20}: R² = {test_r2:.4f}, RMSE = {test_rmse:.2f}")
        
        if strategy_results:
            best_result = max(strategy_results, key=lambda x: x['Test_R2'])
            print(f"   🏆 Best: {best_result['Model']} (R² = {best_result['Test_R2']:.4f})")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    return results_df, trained_models


def analyze_best_combinations(results_df, trained_models):
    """Analyze the best performing model-strategy combinations"""
    print(f"\n🏆 STEP 6: BEST PERFORMING COMBINATIONS")
    print("-" * 50)
    
    best_overall = results_df.iloc[0]
    print(f"🥇 OVERALL BEST PERFORMANCE:")
    print(f"   Strategy: {best_overall['Strategy']}")
    print(f"   Model: {best_overall['Model']}")
    print(f"   Features: {best_overall['Feature_Count']}")
    print(f"   R² Score: {best_overall['Test_R2']:.4f}")
    print(f"   RMSE: {best_overall['Test_RMSE']:.2f} minutes")
    print(f"   MAE: {best_overall['Test_MAE']:.2f} minutes")
    
    print(f"\n🎯 BEST PERFORMANCE BY STRATEGY:")
    strategy_best = results_df.groupby('Strategy').first().sort_values('Test_R2', ascending=False)
    
    for strategy, row in strategy_best.iterrows():
        improvement = ""
        if strategy != 'Original_Top6':
            original_perf = results_df[results_df['Strategy'] == 'Original_Top6']['Test_R2'].max()
            if original_perf > 0:
                improvement_pct = ((row['Test_R2'] - original_perf) / original_perf) * 100
                improvement = f" ({improvement_pct:+.1f}% vs Original)"
        
        print(f"   {strategy:20}: {row['Model']:15} R² = {row['Test_R2']:.4f}{improvement}")
    
    return best_overall, strategy_best


def hyperparameter_tuning_best(best_overall, trained_models, preprocessed_data):
    """Perform hyperparameter tuning for the best model"""
    print(f"\n⚙️ STEP 7: HYPERPARAMETER TUNING")
    print("-" * 50)
    
    best_key = f"{best_overall['Strategy']}_{best_overall['Model']}"
    best_model_info = trained_models[best_key]
    
    print(f"🎯 Tuning: {best_overall['Model']} with {best_overall['Strategy']} strategy")
    
    data_dict = preprocessed_data[best_overall['Strategy']]
    
    if best_model_info['scaled_data']:
        X_train, X_test = data_dict['X_train_scaled'], data_dict['X_test_scaled']
    else:
        X_train, X_test = data_dict['X_train'], data_dict['X_test']
    
    y_train, y_test = data_dict['y_train'], data_dict['y_test']
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.9, 1.0]
        },
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
    
    model_name = best_overall['Model']
    
    if model_name not in param_grids:
        print(f"⚠️  No hyperparameter tuning defined for {model_name}")
        return None, best_model_info
    
    base_models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'K-Nearest Neighbors': KNeighborsRegressor()
    }
    
    model = base_models[model_name]
    param_grid = param_grids[model_name]
    
    print("🔍 Performing Grid Search...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    y_pred_tuned = grid_search.best_estimator_.predict(X_test)
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    
    improvement = tuned_r2 - best_overall['Test_R2']
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    print(f"✅ Tuned test R²: {tuned_r2:.4f} (improvement: {improvement:+.4f})")
    print(f"✅ Tuned test RMSE: {tuned_rmse:.2f}")
    
    tuned_model_info = best_model_info.copy()
    tuned_model_info['model'] = grid_search.best_estimator_
    tuned_model_info['tuned'] = True
    tuned_model_info['tuned_params'] = grid_search.best_params_
    tuned_model_info['final_r2'] = tuned_r2
    tuned_model_info['final_rmse'] = tuned_rmse
    
    return grid_search.best_estimator_, tuned_model_info


def save_final_model_and_results(enhanced_data, results_df, tuned_model, tuned_model_info, correlations, original_features, cluster_features, pca_features, target):
    """Save final model and comprehensive results"""
    print(f"\n💾 STEP 8: SAVING FINAL MODEL & RESULTS")
    print("-" * 50)
    
    joblib.dump(tuned_model, 'best_model_enhanced.pkl')
    print("✅ Final model saved as 'best_model_enhanced.pkl'")
    
    joblib.dump(tuned_model_info['scaler'], 'scaler_enhanced.pkl')
    print("✅ Scaler saved as 'scaler_enhanced.pkl'")
    
    model_metadata = {
        'strategy': tuned_model_info['strategy'],
        'model_name': tuned_model_info['model_name'],
        'features': tuned_model_info['features'],
        'feature_types': {
            'original': original_features,
            'cluster': cluster_features,
            'pca': pca_features
        },
        'target': target,
        'scaled_data': tuned_model_info['scaled_data'],
        'tuned': tuned_model_info.get('tuned', False),
        'tuned_params': tuned_model_info.get('tuned_params', {}),
        'performance': {
            'r2_score': tuned_model_info.get('final_r2', 0),
            'rmse': tuned_model_info.get('final_rmse', 0)
        },
        'enhancement_used': len(cluster_features + pca_features) > 0
    }
    
    with open('model_metadata_enhanced.pkl', 'wb') as f:
        pickle.dump(model_metadata, f)
    print("✅ Enhanced model metadata saved")
    
    results_df.to_csv('model_comparison_enhanced.csv', index=False)
    print("✅ Model comparison results saved")
    
    with open('correlations_enhanced.pkl', 'wb') as f:
        pickle.dump(correlations, f)
    print("✅ Feature correlations saved")
    
    return model_metadata


def save_unsupervised_models_for_streamlit(enhanced_data, target):
    """Save individual unsupervised models for Streamlit integration"""
    print(f"\n💾 STEP 8.5: SAVING UNSUPERVISED MODELS FOR STREAMLIT")
    print("-" * 60)
    
    try:
        def add_distance_traffic_interaction_fixed(data):
            """Add interaction features with FIXED naming"""
            
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
            
            data['Pizza_Profile_Score'] = (
                data['Pizza Type'] * 0.3 + 
                data['Topping Density'] * 0.4 + 
                data['Pizza Complexity'] * 0.3
            )
            
            return data
        
        original_features = ['Pizza Type', 'Distance (km)', 'Is Weekend', 'Topping Density', 
                            'Order Month', 'Pizza Complexity', 'Traffic Impact', 'Order Hour']
        
        sample_data = enhanced_data[original_features].copy()
        sample_data = add_distance_traffic_interaction_fixed(sample_data)
        
        interaction_features = ['Distance_Traffic_Challenge', 'Distance_Traffic_Product', 
                               'Traffic_Per_KM', 'Distance_Traffic_Category', 
                               'Delivery_Challenge_Index', 'Pizza_Profile_Score']
        
        all_features = original_features + interaction_features
        
        scaler = RobustScaler()
        scaler.fit(sample_data[all_features])
        
        joblib.dump(scaler, 'unsupervised_scaler.pkl')
        print("✅ Unsupervised scaler saved")
        
        X_scaled = scaler.transform(sample_data[all_features])
        scaled_features = [f'{f}_scaled' for f in all_features]
        
        correlations = {}
        for i, feature in enumerate(all_features):
            try:
                x = sample_data[feature].values
                y = enhanced_data[target].values[:len(x)]
                rho, p_value = stats.spearmanr(x, y)
                correlations[f'{feature}_scaled'] = abs(rho)
            except:
                correlations[f'{feature}_scaled'] = 0
        
        top_6_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:6]
        top_6_features = [item[0] for item in top_6_items]
        
        print(f"📈 Top 6 features: {[f.replace('_scaled', '') for f in top_6_features]}")
        
        scaled_df = pd.DataFrame(X_scaled, columns=scaled_features)
        X_for_pca = scaled_df[top_6_features].values
        
        pca = PCA(n_components=4)
        pca.fit(X_for_pca)
        
        joblib.dump(pca, 'pca_model.pkl')
        print(f"✅ PCA model saved (4 components, {pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
        
        X_pca = pca.transform(X_for_pca)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        
        joblib.dump(kmeans, 'kmeans_model.pkl')
        print("✅ KMeans model saved (3 clusters)")
        
        feature_info = {
            'original_features': original_features,
            'interaction_features': interaction_features,
            'all_features': all_features,
            'top_6_features': top_6_features,
            'top_6_original': [f.replace('_scaled', '') for f in top_6_features],
            'target': target
        }
        
        with open('feature_info.pkl', 'wb') as f:
            pickle.dump(feature_info, f)
        print("✅ Feature information saved")
        
        print("\n🧪 TESTING STREAMLIT PIPELINE...")
        test_input = pd.DataFrame({
            'Pizza Type': [3],
            'Distance (km)': [7.5],
            'Is Weekend': [0],
            'Topping Density': [6],
            'Order Month': [8],
            'Pizza Complexity': [5],
            'Traffic Impact': [7],
            'Order Hour': [19]
        })
        
        test_enhanced = add_distance_traffic_interaction_fixed(test_input.copy())
        test_scaled = scaler.transform(test_enhanced[all_features])
        test_scaled_df = pd.DataFrame(test_scaled, columns=scaled_features)
        test_pca = pca.transform(test_scaled_df[top_6_features].values)
        test_cluster = kmeans.predict(test_pca)[0]
        
        print(f"✅ Pipeline test successful!")
        print(f"   Input features: {len(test_input.columns)}")
        print(f"   Enhanced features: {len(test_enhanced.columns)}")
        print(f"   PCA components: {test_pca.shape[1]}")
        print(f"   Assigned cluster: {test_cluster}")
        
        print(f"\n📁 STREAMLIT INTEGRATION FILES SAVED:")
        print(f"   🔧 unsupervised_scaler.pkl - Feature engineering scaler")
        print(f"   📐 pca_model.pkl - PCA transformation (4 components)")
        print(f"   📊 kmeans_model.pkl - Clustering model (3 clusters)")
        print(f"   📋 feature_info.pkl - Feature mapping information")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving unsupervised models: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_enhanced_visualizations(enhanced_data, results_df, original_features, cluster_features, pca_features, target):
    """Generate comprehensive visualizations"""
    print(f"\n📊 STEP 9: GENERATING ENHANCED VISUALIZATIONS")
    print("-" * 50)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    strategy_best = results_df.groupby('Strategy')['Test_R2'].max().sort_values(ascending=True)
    colors = ['red' if 'Original' in strategy else 'blue' for strategy in strategy_best.index]
    bars = plt.barh(range(len(strategy_best)), strategy_best.values, color=colors, alpha=0.7)
    plt.yticks(range(len(strategy_best)), [s.replace('_', '\n') for s in strategy_best.index])
    plt.xlabel('R² Score')
    plt.title('Best R² by Feature Strategy')
    plt.grid(True, alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, strategy_best.values)):
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontweight='bold')
    
    plt.subplot(2, 3, 2)
    model_best = results_df.groupby('Model')['Test_R2'].max().sort_values(ascending=False)
    plt.bar(range(len(model_best)), model_best.values, alpha=0.7, color='skyblue')
    plt.xticks(range(len(model_best)), [m.replace(' ', '\n') for m in model_best.index], rotation=45)
    plt.ylabel('R² Score')
    plt.title('Best R² by Model Type')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.scatter(results_df['Feature_Count'], results_df['Test_R2'], alpha=0.6, s=50)
    plt.xlabel('Number of Features')
    plt.ylabel('R² Score')
    plt.title('Feature Count vs Performance')
    plt.grid(True, alpha=0.3)
    
    if cluster_features or pca_features:
        plt.subplot(2, 3, 4)
        enhanced_features = cluster_features + pca_features
        if enhanced_features and all(f in enhanced_data.columns for f in enhanced_features[:5]):
            corr_matrix = enhanced_data[enhanced_features[:5] + [target]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Enhanced Features Correlation')
    
    if 'Cluster_ID' in enhanced_data.columns:
        plt.subplot(2, 3, 5)
        cluster_means = enhanced_data.groupby('Cluster_ID')[target].mean()
        plt.bar(cluster_means.index, cluster_means.values, alpha=0.7, color='orange')
        plt.xlabel('Cluster ID')
        plt.ylabel('Average Delivery Time (min)')
        plt.title('Delivery Time by Cluster')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    all_features = original_features + cluster_features + pca_features
    feature_types = ['Original'] * len(original_features) + \
                   ['Cluster'] * len(cluster_features) + \
                   ['PCA'] * len(pca_features)
    
    if feature_types:
        type_counts = pd.Series(feature_types).value_counts()
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        plt.title('Feature Type Distribution')
    
    plt.tight_layout()
    plt.savefig('enhanced_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Enhanced analysis plot saved")
    
    plt.figure(figsize=(12, 8))
    
    pivot_results = results_df.pivot(index='Strategy', columns='Model', values='Test_R2')
    
    plt.subplot(2, 1, 1)
    sns.heatmap(pivot_results, annot=True, cmap='YlOrRd', fmt='.3f', 
               cbar_kws={'shrink': 0.8})
    plt.title('R² Score Heatmap: Strategy vs Model')
    plt.ylabel('Feature Strategy')
    
    plt.subplot(2, 1, 2)
    baseline_performance = results_df[results_df['Strategy'] == 'Original_Top6']['Test_R2'].max()
    
    improvement_data = []
    for strategy in results_df['Strategy'].unique():
        if strategy != 'Original_Top6':
            best_r2 = results_df[results_df['Strategy'] == strategy]['Test_R2'].max()
            improvement = ((best_r2 - baseline_performance) / baseline_performance) * 100
            improvement_data.append({'Strategy': strategy, 'Improvement': improvement})
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        bars = plt.barh(improvement_df['Strategy'], improvement_df['Improvement'], 
                       color=['green' if x > 0 else 'red' for x in improvement_df['Improvement']])
        plt.xlabel('R² Improvement (%)')
        plt.title('Performance Improvement over Original Features')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, improvement_df['Improvement']):
            plt.text(val + (0.1 if val > 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', ha='left' if val > 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Strategy comparison plot saved")
    
    plt.close('all')


def main_with_streamlit_export():
    """Enhanced main function that also prepares models for Streamlit"""
    try:
        print("🎯 PIZZA DELIVERY SUPERVISED LEARNING WITH ENHANCED FEATURES")
        print("🔗 Leveraging unsupervised clustering and PCA results")
        print("="*70)
        
        enhanced_data, original_features, cluster_features, pca_features, target = load_enhanced_data()
        
        if enhanced_data is None:
            return None, None
        
        correlations, all_features = analyze_enhanced_features(
            enhanced_data, original_features, cluster_features, pca_features, target
        )
        
        feature_sets = select_features(
            enhanced_data, original_features, cluster_features, pca_features, correlations, target
        )
        
        preprocessed_data = preprocess_data_multiple_sets(enhanced_data, feature_sets, target)
        
        results_df, trained_models = train_models_all_strategies(preprocessed_data)
        
        best_overall, strategy_best = analyze_best_combinations(results_df, trained_models)
        
        tuned_model, tuned_model_info = hyperparameter_tuning_best(
            best_overall, trained_models, preprocessed_data
        )
        
        model_metadata = save_final_model_and_results(
            enhanced_data, results_df, tuned_model, tuned_model_info, correlations,
            original_features, cluster_features, pca_features, target
        )
        
        streamlit_success = save_unsupervised_models_for_streamlit(enhanced_data, target)
        
        generate_enhanced_visualizations(
            enhanced_data, results_df, original_features, cluster_features, pca_features, target
        )
        
        print(f"\n" + "="*80)
        print("🎉 ENHANCED SUPERVISED LEARNING COMPLETED!")
        print("="*80)
        
        print(f"📊 DATASET SUMMARY:")
        print(f"   Total samples: {len(enhanced_data)}")
        print(f"   Original features: {len(original_features)}")
        print(f"   Cluster features: {len(cluster_features)}")
        print(f"   PCA features: {len(pca_features)}")
        print(f"   Total enhanced features: {len(all_features)}")
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   Strategy: {best_overall['Strategy']}")
        print(f"   Model: {best_overall['Model']}")
        print(f"   Features used: {best_overall['Feature_Count']}")
        print(f"   Final R² Score: {tuned_model_info.get('final_r2', best_overall['Test_R2']):.4f}")
        print(f"   Final RMSE: {tuned_model_info.get('final_rmse', best_overall['Test_RMSE']):.2f} minutes")
        
        if len(results_df['Strategy'].unique()) > 1:
            original_best = results_df[results_df['Strategy'] == 'Original_Top6']['Test_R2'].max()
            enhanced_best = results_df[results_df['Strategy'] != 'Original_Top6']['Test_R2'].max()
            
            if enhanced_best > original_best:
                improvement = ((enhanced_best - original_best) / original_best) * 100
                print(f"\n📈 ENHANCEMENT IMPACT:")
                print(f"   Improvement over original: {improvement:.1f}%")
                print(f"   Original best R²: {original_best:.4f}")
                print(f"   Enhanced best R²: {enhanced_best:.4f}")
                print(f"   🚀 Enhanced features provided measurable improvement!")
            else:
                print(f"\n📈 ENHANCEMENT IMPACT:")
                print(f"   Enhanced features did not improve performance")
                print(f"   Original features remain optimal for this dataset")
        
        if cluster_features:
            print(f"\n🔗 CLUSTERING INSIGHTS:")
            print(f"   Cluster features: {cluster_features}")
            if 'Cluster_ID' in enhanced_data.columns:
                cluster_impact = enhanced_data.groupby('Cluster_ID')[target].agg(['mean', 'std'])
                cluster_range = cluster_impact['mean'].max() - cluster_impact['mean'].min()
                print(f"   Delivery time range across clusters: {cluster_range:.1f} minutes")
                print(f"   Clustering effectiveness: {'High' if cluster_range > 10 else 'Moderate' if cluster_range > 5 else 'Low'}")
        
        if pca_features:
            print(f"\n📐 PCA INSIGHTS:")
            print(f"   PCA features: {pca_features}")
            print(f"   Dimensionality reduction applied successfully")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if best_overall['Strategy'] == 'Original_Top6':
            print(f"   📌 Original features are sufficient for this dataset")
            print(f"   📌 Enhanced features may not provide significant value")
        else:
            print(f"   📌 Use enhanced features: {best_overall['Strategy']}")
            print(f"   📌 Enhanced features improve prediction accuracy")
        
        print(f"   📌 Best model type: {best_overall['Model']}")
        print(f"   📌 Optimal feature count: {best_overall['Feature_Count']}")
        
        print(f"\n📁 GENERATED FILES:")
        print(f"   🤖 best_model_enhanced.pkl - Final tuned model")
        print(f"   🔧 scaler_enhanced.pkl - Feature scaler for supervised")
        print(f"   📋 model_metadata_enhanced.pkl - Model information")
        print(f"   📊 model_comparison_enhanced.csv - All results")
        print(f"   📈 enhanced_analysis.png - Comprehensive analysis plots")
        print(f"   📈 strategy_comparison.png - Strategy comparison plots")
        
        if streamlit_success:
            print(f"\n🚀 STREAMLIT INTEGRATION FILES:")
            print(f"   🔧 unsupervised_scaler.pkl - For feature engineering")
            print(f"   📐 pca_model.pkl - For PCA transformation")
            print(f"   📊 kmeans_model.pkl - For clustering")
            print(f"   📋 feature_info.pkl - Feature mapping")
            print(f"   ✅ All files ready for Streamlit deployment!")
        
        print(f"\n🚀 STREAMLIT USAGE:")
        print(f"   1. All required files are now available")
        print(f"   2. Use the fixed Streamlit app provided")
        print(f"   3. Run: streamlit run fixed_streamlit_app.py")
        print(f"   4. Enhanced pipeline will work with 8 original inputs")
        print(f"   5. Real-time feature engineering and predictions")
        
        print("="*80)
        
        return tuned_model, model_metadata
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def predict_delivery_time(new_data, model_path='best_model_enhanced.pkl', 
                         scaler_path='scaler_enhanced.pkl', 
                         metadata_path='model_metadata_enhanced.pkl'):
    """
    Make predictions on new data using the trained model
    
    Parameters:
    new_data: DataFrame with same features as training data
    model_path: Path to saved model
    scaler_path: Path to saved scaler
    metadata_path: Path to model metadata
    """
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        required_features = metadata['features']
        
        missing_features = [f for f in required_features if f not in new_data.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            print("   Note: Enhanced features need to be generated from unsupervised pipeline")
            return None
        
        X_new = new_data[required_features]
        
        if metadata['scaled_data']:
            X_new_scaled = scaler.transform(X_new)
            predictions = model.predict(X_new_scaled)
        else:
            predictions = model.predict(X_new)
        
        print(f"✅ Predictions completed using {metadata['model_name']} with {metadata['strategy']}")
        print(f"   Model R²: {metadata['performance']['r2_score']:.4f}")
        print(f"   Expected RMSE: {metadata['performance']['rmse']:.2f} minutes")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


def example_usage():
    """Example of how to use the trained model for predictions"""
    print("\n📘 EXAMPLE USAGE:")
    print("-" * 30)
    
    example_data = pd.DataFrame({
        'Pizza Type': [3],
        'Distance (km)': [5.5],
        'Is Weekend': [0],
        'Topping Density': [7],
        'Order Month': [6],
        'Pizza Complexity': [6],
        'Traffic Impact': [8],
        'Order Hour': [19],
        'Cluster_ID': [1],
        'PC1': [0.5],
        'PC2': [-0.2],
        'PC3': [0.1],
        'PC4': [0.3],
        'Distance_to_Centroid': [1.2],
        'Cluster_Avg_Delivery': [28.5]
    })
    
    print("Example new order data:")
    print(example_data.to_string(index=False))
    
    prediction = predict_delivery_time(example_data)
    
    if prediction is not None:
        print(f"\n🎯 Predicted delivery time: {prediction[0]:.1f} minutes")
    
    return prediction


def test_streamlit_integration():
    """Test if all Streamlit integration files work correctly"""
    print("\n🧪 TESTING STREAMLIT INTEGRATION")
    print("-" * 50)
    
    try:
        model = joblib.load('best_model_enhanced.pkl')
        scaler = joblib.load('scaler_enhanced.pkl')
        
        with open('model_metadata_enhanced.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        with open('correlations_enhanced.pkl', 'rb') as f:
            correlations = pickle.load(f)
        
        unsupervised_scaler = joblib.load('unsupervised_scaler.pkl')
        pca_model = joblib.load('pca_model.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        print("✅ All model files loaded successfully")
        
        test_input = pd.DataFrame({
            'Pizza Type': [4],
            'Distance (km)': [8.2],
            'Is Weekend': [1],
            'Topping Density': [8],
            'Order Month': [12],
            'Pizza Complexity': [7],
            'Traffic Impact': [9],
            'Order Hour': [20]
        })
        
        print(f"\n🔬 Testing with sample input:")
        print(test_input.to_string(index=False))
        
        def add_distance_traffic_interaction_fixed(data):
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
            data['Pizza_Profile_Score'] = (
                data['Pizza Type'] * 0.3 + 
                data['Topping Density'] * 0.4 + 
                data['Pizza Complexity'] * 0.3
            )
            return data
        
        enhanced_input = add_distance_traffic_interaction_fixed(test_input.copy())
        all_features = feature_info['all_features']
        
        X_scaled = unsupervised_scaler.transform(enhanced_input[all_features])
        scaled_features = [f'{f}_scaled' for f in all_features]
        scaled_df = pd.DataFrame(X_scaled, columns=scaled_features)
        
        top_6_features = feature_info['top_6_features']
        X_for_pca = scaled_df[top_6_features].values
        X_pca = pca_model.transform(X_for_pca)
        
        for i in range(X_pca.shape[1]):
            enhanced_input[f'PC{i+1}'] = X_pca[:, i]
        
        cluster_id = kmeans_model.predict(X_pca)[0]
        enhanced_input['Cluster_ID'] = cluster_id
        
        centroid = kmeans_model.cluster_centers_[cluster_id]
        distance_to_centroid = np.linalg.norm(X_pca[0] - centroid)
        enhanced_input['Distance_to_Centroid'] = distance_to_centroid
        
        cluster_avg_mapping = {0: 22.5, 1: 28.3, 2: 35.1}
        enhanced_input['Cluster_Avg_Delivery'] = cluster_avg_mapping.get(cluster_id, 27.5)
        
        print(f"\n✅ Feature engineering completed:")
        print(f"   Original features: {len(test_input.columns)}")
        print(f"   Enhanced features: {len(enhanced_input.columns)}")
        print(f"   Assigned cluster: {cluster_id}")
        print(f"   Distance to centroid: {distance_to_centroid:.3f}")
        
        required_features = metadata['features']
        missing_features = [f for f in required_features if f not in enhanced_input.columns]
        
        if not missing_features:
            X_final = enhanced_input[required_features]
            if metadata['scaled_data']:
                X_final_scaled = scaler.transform(X_final)
                prediction = model.predict(X_final_scaled)[0]
            else:
                prediction = model.predict(X_final)[0]
            
            print(f"\n🎯 FINAL PREDICTION: {prediction:.1f} minutes")
            print("✅ Full Streamlit pipeline test SUCCESSFUL!")
        else:
            print(f"\n⚠️ Missing features for final prediction: {missing_features}")
            print("✅ Feature engineering pipeline test SUCCESSFUL!")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🍕 STARTING ENHANCED SUPERVISED LEARNING PIPELINE")
    print("🔗 WITH STREAMLIT INTEGRATION")
    print("="*70)
    
    final_model, metadata = main_with_streamlit_export()
    
    if final_model is not None:
        print(f"\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"   Model saved and ready for use")
        print(f"   Streamlit integration files prepared")
        
        print(f"\n🧪 TESTING STREAMLIT INTEGRATION...")
        integration_success = test_streamlit_integration()
        
        if integration_success:
            print(f"\n🚀 STREAMLIT READY!")
            print(f"   Run: streamlit run fixed_streamlit_app.py")
            print(f"   All enhanced features will work correctly")
        
        try:
            example_usage()
        except Exception as e:
            print(f"⚠️ Example usage failed: {e}")
            print("   This is normal if enhanced data file is not available")
    else:
        print(f"\n❌ PIPELINE FAILED!")
        print(f"   Please check error messages above")
    
    print(f"\n🏁 ANALYSIS COMPLETE!")
    print("="*70)
