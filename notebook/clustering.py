# ============================================================
# PROJECT: Career Progression & Promotion Gap Analysis
# Company: Palo Alto Networks
# Step 3: K-Means Clustering (ML)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────
# 1. LOAD PROCESSED DATASET
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Processed Dataset")
print("=" * 60)

df = pd.read_csv('hr_data_processed.csv')
print(f"✅ Loaded! Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ──────────────────────────────────────────────
# 2. SELECT FEATURES FOR CLUSTERING
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Selecting Clustering Features")
print("=" * 60)

cluster_features = [
    'Promotion_Gap_Ratio',
    'Role_Stagnation_Index',
    'Training_Intensity_Score',
    'Manager_Stability_Indicator',
    'Career_Growth_Rate',
    'YearsSinceLastPromotion',
    'YearsInCurrentRole',
    'JobLevel',
    'PercentSalaryHike',
    'TrainingTimesLastYear'
]

X = df[cluster_features].copy()
print(f"✅ Features selected: {len(cluster_features)}")
for f in cluster_features:
    print(f"   - {f}")

# ──────────────────────────────────────────────
# 3. NORMALIZE / SCALE DATA
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Normalizing Data (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Data normalized successfully!")

# ──────────────────────────────────────────────
# 4. ELBOW METHOD — Find Best K
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Elbow Method — Finding Best K")
print("=" * 60)

inertia_values = []
silhouette_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_values.append(sil_score)
    print(f"   K={k} → Inertia: {kmeans.inertia_:.1f} | Silhouette: {sil_score:.3f}")

best_k = k_range[silhouette_values.index(max(silhouette_values))]
print(f"\n✅ Best K = {best_k} (highest silhouette score: {max(silhouette_values):.3f})")

# ──────────────────────────────────────────────
# 5. FINAL K-MEANS MODEL
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 5: Training Final K-Means (K={best_k})")
print("=" * 60)

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)
print(f"✅ Clustering Done! {best_k} clusters created.")

cluster_counts = df['Cluster'].value_counts().sort_index()
print("\n📊 Cluster Distribution:")
for cluster, count in cluster_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"   Cluster {cluster}: {count:4d} employees ({pct:.1f}%) {bar}")

# ──────────────────────────────────────────────
# 6. CLUSTER INTERPRETATION & LABELING
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Cluster Interpretation & Labeling")
print("=" * 60)

cluster_summary = df.groupby('Cluster')[cluster_features + ['Attrition']].mean().round(3)
print("\n📋 Cluster Averages:")
print(cluster_summary.to_string())

# Auto-label clusters based on their characteristics
def label_cluster(row):
    promo_gap   = row['Promotion_Gap_Ratio']
    stagnation  = row['Role_Stagnation_Index']
    growth      = row['Career_Growth_Rate']
    training    = row['Training_Intensity_Score']
    manager_stab= row['Manager_Stability_Indicator']

    if growth > cluster_summary['Career_Growth_Rate'].mean() and promo_gap < cluster_summary['Promotion_Gap_Ratio'].mean():
        return '🚀 Fast-Track Performers'
    elif promo_gap > 0.5 and stagnation > 0.5:
        return '🔴 High-Risk Stagnation'
    elif stagnation > cluster_summary['Role_Stagnation_Index'].mean() and growth < cluster_summary['Career_Growth_Rate'].mean():
        return '🟠 Promotion-Stalled'
    elif manager_stab > cluster_summary['Manager_Stability_Indicator'].mean() and training > cluster_summary['Training_Intensity_Score'].mean():
        return '🟢 Stable Long-Term Contributors'
    else:
        return '🟡 Early-Career Explorers'

cluster_labels = {}
for cluster_id in range(best_k):
    row = cluster_summary.loc[cluster_id]
    cluster_labels[cluster_id] = label_cluster(row)

print("\n🏷️  Cluster Labels:")
for cid, label in cluster_labels.items():
    count = cluster_counts.get(cid, 0)
    attrition_rate = cluster_summary.loc[cid, 'Attrition'] * 100
    print(f"   Cluster {cid} → {label}")
    print(f"             Employees: {count} | Attrition Rate: {attrition_rate:.1f}%")

df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

# ──────────────────────────────────────────────
# 7. HIERARCHICAL CLUSTERING (Validation)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Hierarchical Clustering (Validation)")
print("=" * 60)

hierarchical = AgglomerativeClustering(n_clusters=best_k)
df['Cluster_Hierarchical'] = hierarchical.fit_predict(X_scaled)

# Compare overlap
overlap = (df['Cluster'] == df['Cluster_Hierarchical']).mean() * 100
print(f"✅ Hierarchical clustering done!")
print(f"   K-Means vs Hierarchical agreement: ~{overlap:.1f}%")
print(f"   (High agreement = clusters are stable & reliable)")

# ──────────────────────────────────────────────
# 8. PCA FOR VISUALIZATION (2D)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: PCA — Reducing to 2D for Visualization")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

explained = pca.explained_variance_ratio_ * 100
print(f"✅ PCA Done!")
print(f"   PC1 explains: {explained[0]:.1f}% variance")
print(f"   PC2 explains: {explained[1]:.1f}% variance")
print(f"   Total explained: {sum(explained):.1f}%")

# ──────────────────────────────────────────────
# 9. VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Creating Cluster Visualizations...")
print("=" * 60)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Palo Alto Networks — Career Path Clustering Results',
             fontsize=15, fontweight='bold')

# Plot 1: Elbow Curve
ax1 = axes[0, 0]
ax1.plot(list(k_range), inertia_values, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax1.set_title('Elbow Method — Finding Best K', fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.legend()

# Plot 2: Silhouette Scores
ax2 = axes[0, 1]
ax2.plot(list(k_range), silhouette_values, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax2.set_title('Silhouette Scores', fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.legend()

# Plot 3: PCA 2D Cluster Plot
ax3 = axes[0, 2]
for i in range(best_k):
    mask = df['Cluster'] == i
    ax3.scatter(df[mask]['PCA1'], df[mask]['PCA2'],
                c=colors[i % len(colors)], label=f'C{i}: {cluster_labels[i][2:15]}',
                alpha=0.6, s=30)
ax3.set_title('Clusters in 2D (PCA)', fontweight='bold')
ax3.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
ax3.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
ax3.legend(fontsize=7)

# Plot 4: Cluster Distribution Bar
ax4 = axes[1, 0]
short_labels = [f'C{i}' for i in range(best_k)]
counts = [cluster_counts.get(i, 0) for i in range(best_k)]
bars = ax4.bar(short_labels, counts,
               color=colors[:best_k], edgecolor='white')
ax4.set_title('Employees per Cluster', fontweight='bold')
ax4.set_ylabel('Count')
for b, v in zip(bars, counts):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+5,
             str(v), ha='center', fontweight='bold')

# Plot 5: Attrition Rate per Cluster
ax5 = axes[1, 1]
atr_cluster = df.groupby('Cluster')['Attrition'].mean() * 100
bars2 = ax5.bar([f'C{i}' for i in atr_cluster.index],
                atr_cluster.values,
                color=colors[:best_k], edgecolor='white')
ax5.set_title('Attrition Rate per Cluster', fontweight='bold')
ax5.set_ylabel('Attrition Rate (%)')
for b, v in zip(bars2, atr_cluster.values):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
             f'{v:.1f}%', ha='center', fontweight='bold')

# Plot 6: Heatmap — Cluster vs Feature Averages
ax6 = axes[1, 2]
heatmap_data = cluster_summary[['Promotion_Gap_Ratio', 'Role_Stagnation_Index',
                                  'Career_Growth_Rate', 'Training_Intensity_Score',
                                  'Manager_Stability_Indicator']]
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
            ax=ax6, linewidths=0.5)
ax6.set_title('Cluster Feature Averages Heatmap', fontweight='bold')
ax6.set_yticklabels([f'C{i}' for i in heatmap_data.index], rotation=0)

plt.tight_layout()
plt.savefig('data/clustering_plots.png', dpi=150, bbox_inches='tight')
print("✅ Plots saved: data/clustering_plots.png")
plt.show()

# ──────────────────────────────────────────────
# 10. SAVE CLUSTERED DATASET
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Saving Clustered Dataset")
print("=" * 60)

df.to_csv('data/hr_data_clustered.csv', index=False)
print(f"✅ Saved: data/hr_data_clustered.csv")
print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")

print("\n" + "=" * 60)
print("🎉 CLUSTERING COMPLETE!")
print("=" * 60)
print(f"\n📊 Final Cluster Summary:")
for cid, label in cluster_labels.items():
    count = cluster_counts.get(cid, 0)
    atr = cluster_summary.loc[cid, 'Attrition'] * 100
    promo = cluster_summary.loc[cid, 'Promotion_Gap_Ratio']
    print(f"\n   {label}")
    print(f"   Employees: {count} | Attrition: {atr:.1f}% | Promo Gap: {promo:.2f}")

print("\n➡️  Next Step: Run risk_scoring.py (Step 4)")
