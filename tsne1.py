import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



df = pd.read_csv('Enter path')


print("(Samples, Features) {}".format(df.iloc[:,1:].shape))
print(df.iloc[:,1:].describe())

df.iloc[:,1:] = (df.iloc[:,1:]-df.iloc[:,1:].min())/(df.iloc[:,1:].max() - df.iloc[:,1:].min())
print(df.iloc[:,1:].describe())

corr = df.iloc[:,:20].corr()
fig, ax = plt.subplots(figsize=(10,10))
plt.title("Correlation")
sns.heatmap(corr, square=True)
plt.show()

for component in df.columns[1:11]:
    sns.FacetGrid(df,  size=3) \
       .map(sns.kdeplot, component) \
       .add_legend()
    plt.show()


X = df.iloc[:,1:].values
y = df.iloc[:,0].values
print("X ", X.shape, ", y ", y.shape)

pca = PCA(n_components=6).fit(X)
X_pca = pca.transform(X)

principal_components = []
samples, features = X_pca.shape
for m in range(1, features+1):
    principal_components.append("Principal Component {}".format(m))
cols = principal_components+["Release Decade"]
df_pca = pd.DataFrame(np.append(X_pca, y.reshape(samples,1), axis=1), columns=cols)
df_pca["Release Decade"] = df_pca["Release Decade"].astype(int)
print("df_pca.shape = ",df_pca.shape)

sns.pairplot(df_pca, hue="Release Decade",x_vars="Principal Component 1",y_vars="Principal Component 2", size=10)
print(sns.pairplot(df_pca, hue="Release Decade",x_vars="Principal Component 1",y_vars="Principal Component 2", size=10))
#plt.show()

tsne_samples = df_pca.shape[0]
tsne = TSNE(n_components=2, verbose=2, perplexity=50, n_iter=1000)
tsne_results = tsne.fit_transform(df_pca.iloc[:tsne_samples,:-1])
#print(tsne_results)

df_tsne = pd.DataFrame(np.append(tsne_results,
                                 df_pca.iloc[:tsne_samples,-1].values.reshape(tsne_results.shape[0],1),
                                 axis=1),
                       columns=["t-SNE Component 1","t-SNE Component 2","Release Decade"])
df_tsne["Release Decade"] = df_tsne["Release Decade"].astype(int)

sns.pairplot(df_tsne,  size=10)
plt.show()

'''for component in df_tsne.columns[:-1]:
    sns.FacetGrid(df_tsne, hue="Release Decade", size=6) \
       .map(sns.kdeplot, component) \
       .add_legend()
    plt.show()'''

