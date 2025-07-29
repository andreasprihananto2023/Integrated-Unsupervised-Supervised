import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🍕 PIZZA DELIVERY - ROBUST SCALING & SPEARMAN CORRELATION")
print("=" * 70)

# Load data
data = pd.read_excel('Train Data.xlsx')
data.columns = data.columns.str.strip()

# Define features and target
features = ['Pizza Type', 'Distance (km)', 
            'Is Weekend', 'Topping Density', 'Order Month', 'Pizza Complexity', 
            'Traffic Impact', 'Order Hour']
target = 'Delivery Duration (min)'

# Clean data
data_clean = data[features + [target]].dropna()
print(f"📊 Data loaded: {len(data_clean)} rows, {len(features)} features")

# =========================================================================
# COMPREHENSIVE SPEARMAN CORRELATION ANALYSIS
# =========================================================================

def comprehensive_spearman_analysis(data_clean, features, target):
    """Comprehensive Spearman correlation analysis"""
    print(f"\n📈 COMPREHENSIVE SPEARMAN CORRELATION ANALYSIS")
    print("-" * 60)
    
    def spearman_manual(x, y):
        """Manual Spearman correlation calculation"""
        x_ranks = stats.rankdata(x)
        y_ranks = stats.rankdata(y)
        
        n = len(x)
        x_mean = np.mean(x_ranks)
        y_mean = np.mean(y_ranks)
        
        numerator = np.sum((x_ranks - x_mean) * (y_ranks - y_mean))
        x_std = np.sqrt(np.sum((x_ranks - x_mean) ** 2))
        y_std = np.sqrt(np.sum((y_ranks - y_mean) ** 2))
        
        if x_std * y_std == 0:
            return 0
        
        return numerator / (x_std * y_std)
    
    # 1. Correlations with target
    print("1. SPEARMAN CORRELATIONS WITH TARGET:")
    target_correlations = {}
    target_details = []
    
    for i, feature in enumerate(features, 1):
        x = data_clean[feature].values
        y = data_clean[target].values
        
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 1:
            # Manual and scipy calculations
            rho_manual = spearman_manual(x_clean, y_clean)
            rho_scipy, p_value = stats.spearmanr(x_clean, y_clean)
            
            target_correlations[feature] = rho_manual
            
            # Determine significance and strength
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            strength = "Strong" if abs(rho_manual) > 0.5 else "Medium" if abs(rho_manual) > 0.3 else "Weak"
            
            target_details.append({
                'Feature': feature,
                'Spearman_Rho': rho_manual,
                'Abs_Spearman_Rho': abs(rho_manual),
                'Scipy_Rho': rho_scipy,
                'P_Value': p_value,
                'Significance': significance,
                'Strength': strength,
                'Sample_Size': len(x_clean),
                'Difference_Manual_Scipy': abs(rho_manual - rho_scipy)
            })
            
            print(f"   {i:2d}. {feature:20}: ρ = {rho_manual:7.4f} (p={p_value:.4f}, {strength}, {significance})")
        else:
            target_correlations[feature] = 0
            target_details.append({
                'Feature': feature,
                'Spearman_Rho': 0,
                'Abs_Spearman_Rho': 0,
                'Scipy_Rho': 0,
                'P_Value': 1.0,
                'Significance': 'No Data',
                'Strength': 'No Data',
                'Sample_Size': 0,
                'Difference_Manual_Scipy': 0
            })
    
    # 2. Feature-to-feature correlations
    print(f"\n2. FEATURE-TO-FEATURE SPEARMAN CORRELATIONS:")
    n_features = len(features)
    feature_correlations = []
    correlation_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                correlation_matrix[i, j] = 1.0
                feature_correlations.append({
                    'Feature_1': features[i],
                    'Feature_2': features[j],
                    'Spearman_Rho': 1.0,
                    'Abs_Spearman_Rho': 1.0,
                    'P_Value': 0.0,
                    'Significance': 'Perfect',
                    'Relationship': 'Self'
                })
            elif i < j:  # Avoid duplicates
                x = data_clean[features[i]].values
                y = data_clean[features[j]].values
                
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                
                if len(x_clean) > 1:
                    rho_manual = spearman_manual(x_clean, y_clean)
                    rho_scipy, p_value = stats.spearmanr(x_clean, y_clean)
                    
                    correlation_matrix[i, j] = rho_manual
                    correlation_matrix[j, i] = rho_manual
                    
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    
                    if abs(rho_manual) > 0.7:
                        relationship = "Very High"
                    elif abs(rho_manual) > 0.5:
                        relationship = "High"
                    elif abs(rho_manual) > 0.3:
                        relationship = "Medium"
                    else:
                        relationship = "Low"
                    
                    feature_correlations.append({
                        'Feature_1': features[i],
                        'Feature_2': features[j],
                        'Spearman_Rho': rho_manual,
                        'Abs_Spearman_Rho': abs(rho_manual),
                        'P_Value': p_value,
                        'Significance': significance,
                        'Relationship': relationship
                    })
                    
                    # Print high correlations
                    if abs(rho_manual) > 0.7:
                        print(f"   HIGH: {features[i]} <-> {features[j]}: ρ = {rho_manual:.4f}")
                else:
                    correlation_matrix[i, j] = 0
                    correlation_matrix[j, i] = 0
    
    # 3. Summary statistics
    all_target_corrs = [detail['Abs_Spearman_Rho'] for detail in target_details if detail['Sample_Size'] > 0]
    all_feature_corrs = [detail['Abs_Spearman_Rho'] for detail in feature_correlations if detail['Relationship'] not in ['Self', 'No Data']]
    
    print(f"\n3. CORRELATION SUMMARY STATISTICS:")
    print(f"   Target correlations - Mean: {np.mean(all_target_corrs):.4f}, Max: {np.max(all_target_corrs):.4f}")
    print(f"   Feature correlations - Mean: {np.mean(all_feature_corrs):.4f}, Max: {np.max(all_feature_corrs):.4f}")
    
    # Select top features
    sorted_target_corrs = sorted(target_details, key=lambda x: x['Abs_Spearman_Rho'], reverse=True)
    top_features = [item['Feature'] for item in sorted_target_corrs[:8]]
    
    print(f"\n4. TOP 8 FEATURES SELECTED:")
    for i, item in enumerate(sorted_target_corrs[:8], 1):
        print(f"   {i}. {item['Feature']:20}: |ρ| = {item['Abs_Spearman_Rho']:.4f}")
    
    return target_correlations, target_details, feature_correlations, correlation_matrix, top_features

# Execute comprehensive Spearman analysis
target_correlations, target_details, feature_correlations, correlation_matrix, top_features = comprehensive_spearman_analysis(
    data_clean, features, target
)

# =========================================================================
# ROBUST SCALING
# =========================================================================

def apply_robust_scaling(data_clean, top_features):
    """Apply RobustScaler to selected features"""
    print(f"\n📊 ROBUST SCALING")
    print("-" * 50)
    
    X = data_clean[top_features].values
    
    print("1. BEFORE SCALING:")
    scaling_before = []
    for i, feature in enumerate(top_features):
        values = X[:, i]
        stats_before = {
            'Feature': feature,
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Range': np.max(values) - np.min(values),
            'Q1': np.percentile(values, 25),
            'Q3': np.percentile(values, 75),
            'IQR': np.percentile(values, 75) - np.percentile(values, 25)
        }
        scaling_before.append(stats_before)
        print(f"   {feature:20}: Mean={stats_before['Mean']:8.2f}, Range={stats_before['Range']:7.2f}")
    
    # Apply RobustScaler
    print(f"\n2. APPLYING ROBUST SCALER:")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    scaling_after = []
    scaling_parameters = []
    
    for i, feature in enumerate(top_features):
        values_scaled = X_scaled[:, i]
        
        # After scaling stats
        stats_after = {
            'Feature': feature,
            'Mean': np.mean(values_scaled),
            'Median': np.median(values_scaled),
            'Std': np.std(values_scaled),
            'Min': np.min(values_scaled),
            'Max': np.max(values_scaled),
            'Range': np.max(values_scaled) - np.min(values_scaled),
            'Q1': np.percentile(values_scaled, 25),
            'Q3': np.percentile(values_scaled, 75),
            'IQR': np.percentile(values_scaled, 75) - np.percentile(values_scaled, 25)
        }
        scaling_after.append(stats_after)
        
        # Scaling parameters
        scaling_param = {
            'Feature': feature,
            'Original_Median': scaler.center_[i],
            'Original_IQR': scaler.scale_[i],
            'Scaled_Feature_Name': f'{feature}_scaled',
            'Scaling_Formula': f'(X - {scaler.center_[i]:.2f}) / {scaler.scale_[i]:.2f}'
        }
        scaling_parameters.append(scaling_param)
        
        print(f"   {feature:20}: Median={scaler.center_[i]:8.2f}, IQR={scaler.scale_[i]:7.2f}")
    
    # Create scaled DataFrame
    data_scaled = data_clean.copy()
    for i, feature in enumerate(top_features):
        data_scaled[f'{feature}_scaled'] = X_scaled[:, i]
    
    return data_scaled, X_scaled, scaler, scaling_before, scaling_after, scaling_parameters

# Execute robust scaling
data_scaled, X_scaled, scaler, scaling_before, scaling_after, scaling_parameters = apply_robust_scaling(
    data_clean, top_features
)

# =========================================================================
# SAVE COMPREHENSIVE RESULTS TO EXCEL
# =========================================================================

def save_comprehensive_results(data_clean, data_scaled, target_details, feature_correlations, 
                              correlation_matrix, scaling_before, scaling_after, scaling_parameters, 
                              features, top_features, target):
    """Save all comprehensive results to Excel"""
    print(f"\n💾 SAVING COMPREHENSIVE RESULTS TO EXCEL")
    print("-" * 60)
    
    with pd.ExcelWriter('Pizza_Comprehensive_Spearman_RobustScaler.xlsx', engine='openpyxl') as writer:
        
        # 1. Original data
        data_clean.to_excel(writer, sheet_name='1_Original_Data', index=False)
        
        # 2. Data with scaled features
        data_scaled.to_excel(writer, sheet_name='2_Data_with_Scaled', index=False)
        
        # 3. Only scaled features
        scaled_features = [f'{f}_scaled' for f in top_features]
        scaled_only = data_scaled[scaled_features + [target]]
        scaled_only.to_excel(writer, sheet_name='3_Scaled_Features_Only', index=False)
        
        # 4. Target correlations (detailed)
        target_df = pd.DataFrame(target_details)
        target_df = target_df.sort_values('Abs_Spearman_Rho', ascending=False)
        target_df.to_excel(writer, sheet_name='4_Target_Correlations', index=False)
        
        # 5. Feature-to-feature correlations
        feature_df = pd.DataFrame(feature_correlations)
        feature_df = feature_df.sort_values('Abs_Spearman_Rho', ascending=False)
        feature_df.to_excel(writer, sheet_name='5_Feature_Correlations', index=False)
        
        # 6. Correlation matrix
        corr_matrix_df = pd.DataFrame(correlation_matrix, index=features, columns=features)
        corr_matrix_df.to_excel(writer, sheet_name='6_Correlation_Matrix', index=True)
        
        # 7. Scaling parameters
        scaling_params_df = pd.DataFrame(scaling_parameters)
        scaling_params_df.to_excel(writer, sheet_name='7_Scaling_Parameters', index=False)
        
        # 8. Before scaling statistics
        scaling_before_df = pd.DataFrame(scaling_before)
        scaling_before_df.to_excel(writer, sheet_name='8_Stats_Before_Scaling', index=False)
        
        # 9. After scaling statistics
        scaling_after_df = pd.DataFrame(scaling_after)
        scaling_after_df.to_excel(writer, sheet_name='9_Stats_After_Scaling', index=False)
        
        # 10. High correlations summary
        high_corr_target = [item for item in target_details if item['Abs_Spearman_Rho'] > 0.5]
        high_corr_features = [item for item in feature_correlations if item['Abs_Spearman_Rho'] > 0.7]
        
        high_correlations = []
        
        # Add high target correlations
        for item in high_corr_target:
            high_correlations.append({
                'Type': 'Target Correlation',
                'Feature_1': item['Feature'],
                'Feature_2': target,
                'Spearman_Rho': item['Spearman_Rho'],
                'Abs_Correlation': item['Abs_Spearman_Rho'],
                'P_Value': item['P_Value'],
                'Significance': item['Significance'],
                'Interpretation': f"{item['Strength']} correlation with delivery time"
            })
        
        # Add high feature correlations
        for item in high_corr_features:
            high_correlations.append({
                'Type': 'Feature-Feature Correlation',
                'Feature_1': item['Feature_1'],
                'Feature_2': item['Feature_2'],
                'Spearman_Rho': item['Spearman_Rho'],
                'Abs_Correlation': item['Abs_Spearman_Rho'],
                'P_Value': item['P_Value'],
                'Significance': item['Significance'],
                'Interpretation': f"{item['Relationship']} correlation between features"
            })
        
        if high_correlations:
            high_corr_df = pd.DataFrame(high_correlations)
            high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
            high_corr_df.to_excel(writer, sheet_name='10_High_Correlations', index=False)
        
        # 11. Summary report
        summary_data = [
            {'Metric': 'Analysis Type', 'Value': 'Spearman Correlation + RobustScaler'},
            {'Metric': 'Total Features', 'Value': len(features)},
            {'Metric': 'Selected Features (Top 8)', 'Value': len(top_features)},
            {'Metric': 'Data Points', 'Value': len(data_clean)},
            {'Metric': 'Target Variable', 'Value': target},
            {'Metric': 'Scaling Method', 'Value': 'RobustScaler (Median + IQR)'},
            {'Metric': 'Correlation Method', 'Value': 'Spearman Rank Correlation'},
            {'Metric': 'Strongest Target Correlation', 'Value': f"{max(target_details, key=lambda x: x['Abs_Spearman_Rho'])['Feature']} ({max(target_details, key=lambda x: x['Abs_Spearman_Rho'])['Abs_Spearman_Rho']:.4f})"},
            {'Metric': 'High Target Correlations (>0.5)', 'Value': len([item for item in target_details if item['Abs_Spearman_Rho'] > 0.5])},
            {'Metric': 'High Feature Correlations (>0.7)', 'Value': len([item for item in feature_correlations if item['Abs_Spearman_Rho'] > 0.7])},
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='11_Summary_Report', index=False)
    
    print("✅ Comprehensive results saved to 'Pizza_Comprehensive_Spearman_RobustScaler.xlsx'")
    print("📊 Sheets created:")
    print("   1. Original_Data - Raw dataset")
    print("   2. Data_with_Scaled - Original + scaled features")
    print("   3. Scaled_Features_Only - Only scaled features")
    print("   4. Target_Correlations - All correlations with delivery time")
    print("   5. Feature_Correlations - All feature-to-feature correlations")
    print("   6. Correlation_Matrix - Full correlation matrix")
    print("   7. Scaling_Parameters - RobustScaler parameters")
    print("   8. Stats_Before_Scaling - Statistics before scaling")
    print("   9. Stats_After_Scaling - Statistics after scaling")
    print("   10. High_Correlations - Summary of strong correlations")
    print("   11. Summary_Report - Analysis overview")

# Save comprehensive results
save_comprehensive_results(
    data_clean, data_scaled, target_details, feature_correlations, 
    correlation_matrix, scaling_before, scaling_after, scaling_parameters,
    features, top_features, target
)

# =========================================================================
# FINAL COMPREHENSIVE SUMMARY
# =========================================================================

print(f"\n" + "="*70)
print("🎉 COMPREHENSIVE SPEARMAN & ROBUST SCALING ANALYSIS COMPLETE!")
print("="*70)

# Show summary statistics
strongest_target = max(target_details, key=lambda x: x['Abs_Spearman_Rho'])
high_target_corrs = [item for item in target_details if item['Abs_Spearman_Rho'] > 0.5]
high_feature_corrs = [item for item in feature_correlations if item['Abs_Spearman_Rho'] > 0.7]

print(f"📊 SPEARMAN CORRELATION RESULTS:")
print(f"   Strongest correlation: {strongest_target['Feature']} (ρ = {strongest_target['Spearman_Rho']:.4f})")
print(f"   High target correlations (>0.5): {len(high_target_corrs)} features")
print(f"   High feature correlations (>0.7): {len(high_feature_corrs)} pairs")

print(f"\n📈 ROBUST SCALING RESULTS:")
print(f"   Features scaled: {len(top_features)}")
print(f"   All features normalized using median and IQR")
print(f"   Outlier-resistant scaling applied")

print(f"\n💾 COMPREHENSIVE OUTPUT:")
print(f"   📁 'Pizza_Comprehensive_Spearman_RobustScaler.xlsx'")
print(f"   📊 11 detailed sheets with complete analysis")
print(f"   🔍 All correlations, statistics, and parameters included")

print("="*70)
print("🚀 READY FOR ADVANCED CLUSTERING ANALYSIS!")
print("="*70)