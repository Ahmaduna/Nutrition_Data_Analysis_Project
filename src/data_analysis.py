import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import missingno as msno

def load_data(filepath):
    df = pd.read_excel(filepath)
    df = df.set_index('NDB_No')
    return df

def visualize_missing_data(df):
    plt.figure(figsize=(12, 6))
    msno.bar(df, color='steelblue', fontsize=12)
    plt.title("Visualisation des valeurs manquantes", fontsize=16, weight='bold')
    plt.show()

    plt.figure(figsize=(12, 6))
    msno.matrix(df, color=(0.27, 0.52, 0.77), fontsize=12, sparkline=False)
    plt.title("Matrice des valeurs manquantes", fontsize=16, weight='bold')
    plt.show()

    plt.figure(figsize=(12, 6))
    msno.heatmap(df, cmap="coolwarm", fontsize=12)
    plt.title("Heatmap des valeurs manquantes", fontsize=16, weight='bold')
    plt.show()

def plot_correlation_matrices(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 10}, 
                fmt=".2f", linewidths=0.5, linecolor='black', cbar_kws={"shrink": .75})
    plt.title("Matrice de corrélation (Pearson)", fontsize=20, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.show()

    corr_matrix_spearman = df.corr(method='spearman')
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix_spearman, annot=True, cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 10}, 
                fmt=".2f", linewidths=0.5, linecolor='black', cbar_kws={"shrink": .75})
    plt.title("Matrice de corrélation (Spearman)", fontsize=20, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.show()

def perform_pca(df):
    X = df.iloc[:, 1:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    comp = pd.DataFrame({
        "Dimension": ["Dim" + str(x + 1) for x in range(X.shape[1])],
        "Valeur propre": pca.explained_variance_,
        "% variance expliquée": np.round(pca.explained_variance_ratio_ * 100),
        "% cum. var. expliquée": np.round(np.cumsum(pca.explained_variance_ratio_) * 100)
    })

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Dimension", y="% variance expliquée", data=comp, palette="viridis", edgecolor='black')
    plt.axhline(y=25, color='grey', linestyle='--', linewidth=2)
    plt.title("Variance expliquée par dimension", fontsize=18, weight='bold')
    plt.xlabel("Dimension", fontsize=14, weight='bold')
    plt.ylabel("% variance expliquée", fontsize=14, weight='bold')
    plt.show()

    return pca, X_pca

def correlation_graph(pca, x_y, features):
    x, y = x_y
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0, 0, pca.components_[x, i], pca.components_[y, i], head_width=0.07, head_length=0.07, width=0.02, color='darkred', alpha=0.8)
        plt.text(pca.components_[x, i] + 0.05, pca.components_[y, i] + 0.05, features[i], fontsize=12, weight='bold', color='darkblue')
    plt.plot([-1, 1], [0, 0], color='black', ls='--', linewidth=1.5)
    plt.plot([0, 0], [-1, 1], color='black', ls='--', linewidth=1.5)
    plt.xlabel(f'F{x+1} ({round(100*pca.explained_variance_ratio_[x],1)}%)', fontsize=14, weight='bold')
    plt.ylabel(f'F{y+1} ({round(100*pca.explained_variance_ratio_[y],1)}%)', fontsize=14, weight='bold')
    plt.title(f"Cercle des corrélations (F{x+1} et F{y+1})", fontsize=16, weight='bold')
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an), color='black', linewidth=1.5)
    plt.axis('equal')
    plt.show()

def main():
    # Load the dataset
    df = load_data("data/USDA_National_Nutrient_DataBase.xlsx")

    # Visualize missing data
    visualize_missing_data(df)

    # Plot correlation matrices
    plot_correlation_matrices(df)

    # Perform PCA
    pca, X_pca = perform_pca(df)

    # Plot correlation circles
    correlation_graph(pca, (0, 1), df.columns[1:])
    correlation_graph(pca, (0, 2), df.columns[1:])
    correlation_graph(pca, (1, 2), df.columns[1:])

    # Visualize individuals in the factorial plane
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='darkgreen', edgecolor='black', alpha=0.6, s=80)
    plt.xlabel(f"Dimension 1 ({round(100*pca.explained_variance_ratio_[0],1)}%)", fontsize=14, weight='bold')
    plt.ylabel(f"Dimension 2 ({round(100*pca.explained_variance_ratio_[1],1)}%)", fontsize=14, weight='bold')
    plt.title("Premier plan factoriel", fontsize=18, weight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main()
