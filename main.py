# tarea3_msd_clustering.py
# Tarea 3 - Clustering y Reducción de Dimensionalidad en Million Song Dataset
# Basado en el laboratorio de Wine Quality (UCI) pero adaptado al enunciado de la tarea.

import os
import glob
import warnings

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score

import umap

warnings.filterwarnings("ignore")

# ==========================
# 1. CONFIGURACIÓN GENERAL
# ==========================

# Cambia esta ruta a donde tengas la carpeta "MillionSongSubset"
DATA_DIR = "/ruta/a/MillionSongSubset"  # <-- EDITAR

MAX_FILES = 10000  # máximo de canciones a cargar (la tarea pide 10.000)
RANDOM_STATE = 42

# Atributos que intentaremos extraer de cada .h5
# (puedes ajustar la lista según lo que tenga tu subset)
ANALYSIS_FEATURES = [
    "tempo",
    "duration",
    "loudness",
    "key",
    "mode",
    "time_signature",
    "danceability",
    "energy"
]

METADATA_FEATURES = [
    "year"
]


# ==========================
# 2. CARGA DEL DATASET MSD
# ==========================

def extract_song_features(h5_path):
    """
    Extrae un subconjunto de features numéricas desde un archivo .h5 del Million Song Dataset.
    Devuelve un diccionario {feature: valor} o None si falla algo.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            row = f["analysis/songs"][0]

            data = {}

            # Features del grupo analysis/songs (casi todas están ahí)
            analysis_names = row.dtype.names
            for feat in ANALYSIS_FEATURES:
                if feat in analysis_names:
                    data[feat] = float(row[feat])
                else:
                    # Si no existe el campo en este subset, lo marcamos como NaN
                    data[feat] = np.nan

            # Metadata (year)
            if "metadata/songs" in f:
                mrow = f["metadata/songs"][0]
                mnames = mrow.dtype.names
                if "year" in mnames:
                    data["year"] = float(mrow["year"])
                else:
                    data["year"] = np.nan
            else:
                data["year"] = np.nan

            return data
    except Exception:
        # En caso de problemas con algún archivo, lo ignoramos
        return None


def load_msd_subset(base_dir, max_files=10000):
    """
    Recorre recursivamente base_dir buscando archivos .h5 y
    genera un DataFrame con las features definidas arriba.
    """
    base_dir = Path(base_dir)
    all_h5 = list(base_dir.rglob("*.h5"))
    if max_files is not None:
        all_h5 = all_h5[:max_files]

    rows = []
    for i, h5_path in enumerate(all_h5, start=1):
        feats = extract_song_features(str(h5_path))
        if feats is not None:
            rows.append(feats)
        if i % 1000 == 0:
            print(f"Loaded {i} files...")

    df = pd.DataFrame(rows)
    print(f"Loaded dataframe shape: {df.shape}")
    return df


# ==========================
# 3. PREPROCESAMIENTO + EDA
# ==========================

def run_eda(df):
    print("\n=== EDA: overview ===")
    print(df.info())
    print("\n=== Missing values per column ===")
    print(df.isna().sum())

    print("\n=== Descriptive statistics ===")
    print(df.describe(include="all"))

    # Correlaciones
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, square=True)
    plt.title("Correlation heatmap (numerical features)")
    plt.tight_layout()
    plt.show()

    # Algunos histogramas y boxplots (tempo, duration, loudness si existen)
    for col in ["tempo", "duration", "loudness"]:
        if col in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            df[col].hist(bins=30, ax=axes[0])
            axes[0].set_title(f"Histogram of {col}")
            axes[0].set_xlabel(col)
            axes[0].set_ylabel("count")

            sns.boxplot(x=df[col], ax=axes[1])
            axes[1].set_title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()


def preprocess(df):
    """
    - Selecciona solo columnas numéricas.
    - Maneja NaNs.
    - Estandariza con StandardScaler.
    - Genera una etiqueta de referencia (y_true) basada en year (por décadas).
    """
    # Eliminamos columnas totalmente vacías
    df = df.dropna(axis=1, how="all")

    # Rellenamos NaNs con la media de cada columna
    df = df.fillna(df.mean(numeric_only=True))

    num_df = df.select_dtypes(include=[np.number]).copy()

    # Etiqueta de referencia: discretizar 'year' por décadas (si existe)
    if "year" in num_df.columns:
        years = num_df["year"].values
        # Algunos registros tienen year=0 => lo tratamos como "sin año"
        years_clean = np.where(years <= 0, np.nan, years)
        # Décadas (ej: 1990–1999, 2000–2009, etc.)
        decade = (years_clean // 10) * 10
        # Reemplazamos NaN por -1 como clase "sin año"
        decade = np.where(np.isnan(decade), -1, decade)
        y_true = decade.astype(int)
        print("Using 'year' decade bins as reference labels for Rand/ARI.")
    else:
        # Si no hay year, creamos una etiqueta ficticia a partir de tempo
        print("No 'year' column found. Using tempo bins as pseudo-labels.")
        if "tempo" in num_df.columns:
            tempo = num_df["tempo"].values
        else:
            # Tomamos cualquier columna numérica para discretizar
            col0 = num_df.columns[0]
            tempo = num_df[col0].values
            print(f"Using {col0} as fallback for pseudo-label binning.")
        y_true = pd.qcut(tempo, q=4, labels=False, duplicates="drop")
        y_true = y_true.astype(int)

    # Eliminamos la columna year del espacio de features, pero dejamos tempo, etc.
    feature_cols = [c for c in num_df.columns if c != "year"]
    X = num_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Feature matrix shape:", X_scaled.shape)
    print("Reference labels shape:", y_true.shape)
    return X_scaled, y_true, feature_cols


# ==========================
# 4. PCA
# ==========================

def run_pca(X_scaled, color_attr_values=None, attr_name="tempo"):
    """
    - PCA 2D y 10D.
    - Reporta varianza explicada acumulada.
    - Devuelve X_pca2 y X_pca10.
    - Hace gráfico 2D coloreado por algún atributo musical.
    """
    print("\n=== PCA ===")

    # PCA 10D
    pca10 = PCA(n_components=10, random_state=RANDOM_STATE)
    X_pca10 = pca10.fit_transform(X_scaled)
    cum_var_10 = np.cumsum(pca10.explained_variance_ratio_)
    print("Explained variance ratio (10D):", pca10.explained_variance_ratio_)
    print("Cumulative explained variance (10D):", cum_var_10)

    # PCA 2D
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca2 = pca2.fit_transform(X_scaled)
    cum_var_2 = np.sum(pca2.explained_variance_ratio_)
    print(f"Cumulative explained variance (2D): {cum_var_2:.3f}")

    # Visualización 2D
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(X_pca2[:, 0], X_pca2[:, 1], s=5)
    axs[0].set_title("PCA 2D (no color)")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")

    if color_attr_values is not None:
        sc = axs[1].scatter(X_pca2[:, 0], X_pca2[:, 1], s=5, c=color_attr_values, cmap="viridis")
        plt.colorbar(sc, ax=axs[1], label=attr_name)
        axs[1].set_title(f"PCA 2D — colored by {attr_name}")
    else:
        axs[1].scatter(X_pca2[:, 0], X_pca2[:, 1], s=5)
        axs[1].set_title("PCA 2D")

    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")
    plt.tight_layout()
    plt.show()

    return X_pca2, X_pca10, pca2, pca10


# ==========================
# 5. UMAP
# ==========================

def run_umap_experiments(X_scaled, X_pca10, color_attr_values=None, attr_name="tempo"):
    """
    Aplica UMAP:
      - Sobre el espacio original escalado.
      - Sobre el espacio PCA 10D.
    Para varias configuraciones de n_neighbors y min_dist.
    Devuelve un diccionario con los embeddings.
    """
    print("\n=== UMAP experiments ===")

    umap_configs = [
        {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2},
        {"n_neighbors": 30, "min_dist": 0.1, "n_components": 2},
        {"n_neighbors": 50, "min_dist": 0.3, "n_components": 2},
    ]

    results = {}

    for space_name, X in [("X_scaled", X_scaled), ("PCA10", X_pca10)]:
        for cfg in umap_configs:
            print(f"Fitting UMAP on {space_name} with {cfg}")
            umap_model = umap.UMAP(
                n_neighbors=cfg["n_neighbors"],
                min_dist=cfg["min_dist"],
                n_components=cfg["n_components"],
                metric="euclidean",
                random_state=RANDOM_STATE
            )
            X_umap = umap_model.fit_transform(X)
            key = f"{space_name}_nn{cfg['n_neighbors']}_md{cfg['min_dist']}"
            results[key] = X_umap

            # Plot 2D
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].scatter(X_umap[:, 0], X_umap[:, 1], s=5)
            axs[0].set_title(f"UMAP 2D on {space_name}\n{cfg}")
            axs[0].set_xlabel("UMAP1")
            axs[0].set_ylabel("UMAP2")

            if color_attr_values is not None:
                sc = axs[1].scatter(X_umap[:, 0], X_umap[:, 1], s=5, c=color_attr_values, cmap="viridis")
                plt.colorbar(sc, ax=axs[1], label=attr_name)
                axs[1].set_title(f"UMAP 2D on {space_name} — colored by {attr_name}")
            else:
                axs[1].scatter(X_umap[:, 0], X_umap[:, 1], s=5)
                axs[1].set_title(f"UMAP 2D on {space_name}")

            axs[1].set_xlabel("UMAP1")
            axs[1].set_ylabel("UMAP2")
            plt.tight_layout()
            plt.show()

    return results


# ==========================
# 6. CLUSTERING
# ==========================

def evaluate_clustering_space(X_emb, y_true, space_name, k_list, eps_list, min_samples_list):
    """
    Ejecuta KMeans y DBSCAN en un espacio dado.
    Devuelve una lista de diccionarios con métricas.
    """
    records = []

    # ---- KMeans con diferentes k ----
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels_km = km.fit_predict(X_emb)
        if len(set(labels_km)) > 1:
            sil = silhouette_score(X_emb, labels_km)
        else:
            sil = np.nan
        ri = rand_score(y_true, labels_km)
        ari = adjusted_rand_score(y_true, labels_km)

        records.append({
            "space": space_name,
            "algorithm": "KMeans",
            "k_or_eps": k,
            "min_samples": np.nan,
            "silhouette": sil,
            "rand_index": ri,
            "adjusted_rand_index": ari
        })

    # ---- DBSCAN con diferentes parámetros ----
    for eps in eps_list:
        for ms in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels_db = db.fit_predict(X_emb)
            unique_lbl = set(labels_db)
            if len(unique_lbl) > 1 and unique_lbl != {-1}:
                sil = silhouette_score(X_emb, labels_db)
            else:
                sil = np.nan
            ri = rand_score(y_true, labels_db)
            ari = adjusted_rand_score(y_true, labels_db)

            records.append({
                "space": space_name,
                "algorithm": "DBSCAN",
                "k_or_eps": eps,
                "min_samples": ms,
                "silhouette": sil,
                "rand_index": ri,
                "adjusted_rand_index": ari
            })

    return records


def run_all_clustering(X_scaled, X_pca2, X_pca10, umap_dict, y_true):
    """
    Aplica KMeans y DBSCAN en:
      - X_scaled
      - X_pca2
      - X_pca10
      - un par de espacios UMAP (por ejemplo, los dos primeros).
    """
    all_records = []

    # Lista de espacios a evaluar
    spaces = [
        ("X_scaled", X_scaled),
        ("PCA2", X_pca2),
        ("PCA10", X_pca10),
    ]

    # Añadimos algunas variantes UMAP (puedes ajustar cuántas)
    for i, (key, X_umap) in enumerate(umap_dict.items()):
        if i >= 2:  # por ejemplo, solo dos configuraciones UMAP para clustering
            break
        spaces.append((f"UMAP_{key}", X_umap))

    # Rango de k para KMeans
    k_list = [3, 4, 5, 6, 8, 10]
    # Parámetros DBSCAN para explorar
    eps_list = [0.5, 1.0, 1.5]
    min_samples_list = [5, 10, 20]

    for name, X_emb in spaces:
        print(f"\n=== Clustering on space: {name} ===")
        recs = evaluate_clustering_space(X_emb, y_true, name, k_list, eps_list, min_samples_list)
        all_records.extend(recs)

    df_metrics = pd.DataFrame(all_records)
    print("\n=== Summary of clustering metrics ===")
    print(df_metrics.sort_values(by=["space", "algorithm", "silhouette"], ascending=[True, True, False]))
    return df_metrics


# ==========================
# 7. MAIN
# ==========================

def main():
    # 1) Carga de datos
    print("Loading Million Song Subset from:", DATA_DIR)
    df = load_msd_subset(DATA_DIR, max_files=MAX_FILES)

    # 2) EDA breve
    run_eda(df)

    # 3) Preprocesamiento + creación de y_true
    X_scaled, y_true, feature_cols = preprocess(df)

    # Para colorear en PCA/UMAP, usamos 'tempo' si existe, si no la primera columna
    if "tempo" in df.columns:
        color_attr = df["tempo"].values
        color_name = "tempo"
    else:
        color_attr = df[feature_cols[0]].values
        color_name = feature_cols[0]
        print(f"Using {color_name} as color attribute.")

    # 4) PCA (2D y 10D)
    X_pca2, X_pca10, pca2, pca10 = run_pca(X_scaled, color_attr_values=color_attr, attr_name=color_name)

    # 5) UMAP en espacio original y PCA10
    umap_results = run_umap_experiments(X_scaled, X_pca10, color_attr_values=color_attr, attr_name=color_name)

    # 6) Clustering (KMeans y DBSCAN) en distintos espacios
    df_metrics = run_all_clustering(X_scaled, X_pca2, X_pca10, umap_results, y_true)

    # 7) Guardar métricas a CSV para usar en el informe
    df_metrics.to_csv("tarea3_clustering_metrics.csv", index=False)
    print("\nMetrics saved to tarea3_clustering_metrics.csv")


if __name__ == "__main__":
    main()
