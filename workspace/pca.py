# Import dependencies
from funcs import load_data, load_labels
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from scipy.spatial import distance
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# PCA 
patient_data = load_data("data/connectomes_cobre_scale_444.npy")

# reshape into 146 [patients] x (444x444) [patient data] 2d array
shaped_data = np.reshape(patient_data, (146, -1))

pca = PCA(n_components=2)
components = pca.fit_transform(shaped_data)

print('PCA data: ')
print(components)

# load labels and convert data to pd dataframe
labels = load_labels("data/subjects.txt")
df = pd.DataFrame(components, columns=['pca_1','pca_2'])
df = pd.concat([df, labels['labels']], axis=1)
print(df)

# plot data
sns.set_theme()
sns.relplot(
    data=df,
    x="pca_1", y="pca_2", hue="labels",
)

# Split into controls and schizophrenia patients

con_mask = [i == 'cont' for i in df['labels']]
con_df = df[con_mask]

sz_mask = [i == 'sz' for i in df['labels']]
sz_df = df[sz_mask]

intralist = []
interlist = []
print((con_df.iloc[0, 0], con_df.iloc[0,1]))

# Euclidean distances between controls
for i in range(len(con_df)):
    p_1 = (con_df.iloc[i, 0], con_df.iloc[i,1])
    for j in range(i+1, len(con_df)):
        p_2 = (con_df.iloc[j,0], con_df.iloc[j,1])
        intralist.append(distance.euclidean(p_1,p_2))
for i in range(len(sz_df)):
    p_1 = (sz_df.iloc[i, 0], sz_df.iloc[i,1])
    for j in range(i+1, len(sz_df)):
        p_2 = (sz_df.iloc[j,0], sz_df.iloc[j,1])
        intralist.append(distance.euclidean(p_1,p_2))

for i in range(len(con_df)):
    con = (con_df.iloc[i, 0], con_df.iloc[i,1])
    for j in range(len(sz_df)):
        sz = (sz_df.iloc[j,0], sz_df.iloc[j,1])
        interlist.append(distance.euclidean(con,sz))

# Plot distributions

fig,ax = plt.subplots(figsize=(15,10))
for l in [interlist,intralist]:
    sns.histplot(ax=ax,data=l, kde=True)


# Kolmogorov smirnov tests

results = ks_2samp(interlist, intralist)

print(results)

plt.show()