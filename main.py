# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
# Numpy arrays are used to store training and test data.
import numpy as np
# Pandas is used to manipulate tabular data.
import pandas as pd
# Matplotlib is used to plot graphs.
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Style options for plots.
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Convenience function to create display a progress bar.
# Source : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

# Saves a figure to a file
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join("./figs", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Import data
df = pd.read_csv("welddb/welddb.data", sep=r"\s+", header=None, engine="python")

# Stats
#print(df.shape)
#print(df.info())
#print(df.describe())

# Replacing "N" value by NaN
df.replace("N", pd.NA, inplace=True)
#print(df.isna().sum())

df = df.apply(pd.to_numeric, errors='ignore')
#print(df.info())

# Delete column with more than 30% NaN value
df_clean = df.loc[:, df.isna().mean() < 0.3]

#print(df_clean.info())

num_cols = df_clean.select_dtypes(include='number').columns
cat_cols = df_clean.select_dtypes(exclude='number').columns

df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
df_clean[cat_cols] = df_clean[cat_cols].fillna(df_clean[cat_cols].mode().iloc[0])

print(df_clean.info())

# Stats
print(df_clean.describe(include='all'))

# Numeric variables distribution
df_clean[num_cols].hist(bins=30, figsize=(15,10))
plt.suptitle("Distribution des variables numériques")
plt.show()

# Categorical variables analysis
for col in cat_cols:
    print(f"\n{col} : {df_clean[col].nunique()} modalités")
    print(df_clean[col].value_counts().head())

# Correlation analysis
corr = df_clean[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[num_cols])

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled[num_cols])

# Affichage de la variance expliquée
print("Variance expliquée par chaque composante :", pca.explained_variance_ratio_)
print("Variance totale expliquée par les 2 composantes :", sum(pca.explained_variance_ratio_))

# Création d'un DataFrame avec les composantes principales
df_pca = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])

# Scatter plot des deux premières composantes
plt.figure(figsize=(8,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.7, edgecolor='k')
plt.xlabel("Composante Principale 1 (52,2% variance)")
plt.ylabel("Composante Principale 2 (32,4% variance)")
plt.title("Scatter plot des 2 premières composantes principales")
plt.grid(True)
plt.show()

# Scree plot : variance expliquée cumulée
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.xlabel("Nombre de composantes principales")
plt.ylabel("Variance expliquée cumulée")
plt.title("Scree plot - Variance expliquée par la PCA")
plt.grid(True)

# Ligne horizontale à 90% pour repérer un seuil
plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.91, "90% variance", color = 'r', fontsize=12)
plt.show()
