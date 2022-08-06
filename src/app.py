# your code here
# Step 1 
# cargar las librerias: 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt #visualization
%matplotlib inline
# Step 2 
# cargamos los datos: 
URL = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
df_raw = pd.read_csv(URL)

# miramos los datos:
df_raw.head(5)

# modificación del dataset:
df = df_raw.loc[:, ["MedInc", "Latitude", "Longitude"]]
df.head()

# Step 3 
# describimos el Dataset:
df.describe()

# Create cluster feature
kmeans = KMeans(n_clusters=6)
df["Cluster"] = kmeans.fit_predict(df)

# Step 4 
# Convert your new 'cluster' column to 'category' type.
df["Cluster"] = df["Cluster"].astype("category")
df.head(5)

# Step 5
# visualización de cluster 

sns.relplot(x="Longitude", y="Latitude", hue="Cluster", data=df,)
plt.title('Visulaización Cluster', size='large');
