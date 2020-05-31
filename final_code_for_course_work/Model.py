import pandas as pd
import numpy as np
from scipy import stats
import patsy as pt
import librosa as lb
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import linear_model

def detecting_outliers(dataframe):
    data = dataframe._get_numeric_data()
    columns = ['Unnamed: 0', 'year', 'popularity', 'LOG_popularity']
    s = data.drop(columns, 1)
    z_scores = stats.zscore(s)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_data = dataframe[filtered_entries]
    return new_data

def linear_dependence(dataframe):
    genres = np.unique(dataframe['top genre'].to_numpy())
    for i in genres:
        new_data = dataframe[dataframe['top genre'] == i]
        new_data = new_data._get_numeric_data()
        if len(new_data)> 20:
            for j in new_data.columns.tolist():
                lst = new_data[j].to_numpy()
                lst1 = new_data['LOG_popularity'].to_numpy()
                plt.scatter(lst, lst1, alpha=0.5)
                plt.xlabel(j)
                plt.ylabel('Popularity of the song')
                plt.show()

def multidimensional_normality(dataframe):
    dict_norm = {}
    data = dataframe._get_numeric_data()
    for i in data.columns.tolist():
        lst = data[i].to_numpy()
        ks_test = stats.kstest(lst, 'norm')
        dict_norm[i] = ks_test
    return dict_norm

def multicollinearity(dataframe):
    data = dataframe._get_numeric_data()
    columns = ['Unnamed: 0', 'year', 'popularity', 'LOG_popularity']
    s = data.drop(columns, 1)
    features = "+".join(s.columns)
    y, X = pt.dmatrices('LOG_popularity ~' + features, data, return_type='dataframe')
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    return vif.round(1)

def autocorrelation(dataframe):
    target_col = dataframe['LOG_popularity']
    columns = ['Unnamed: 0', 'year', 'popularity', 'LOG_popularity','name','artists','top genre']
    data = dataframe.drop(columns, 1)
    X_train, X_test, y_train, y_test = train_test_split(data, target_col,test_size=0.30,random_state=42)
    X_with_constant = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_with_constant)
    results = model.fit()
    y_pred = results.predict(X_with_constant)
    residual = y_train - y_pred
    residual_df = pd.DataFrame(residual, columns=["ei"]).reset_index(drop=True)
    residual_df['ei_square'] = np.square(residual_df['ei'])
    sum_of_squared_residuals = residual_df.sum()["ei_square"]
    residual_df['ei_minus_1'] = residual_df['ei'].shift()
    residual_df.dropna(inplace=True)
    residual_df['ei_sub_ei_minus_1'] = residual_df['ei'] - residual_df['ei_minus_1']
    residual_df['square_of_ei_sub_ei_minus_1'] = np.square(residual_df['ei_sub_ei_minus_1'])
    sum_of_squared_of_difference_residuals = residual_df.sum()["square_of_ei_sub_ei_minus_1"]
    res = sum_of_squared_of_difference_residuals / sum_of_squared_residuals
    sm.graphics.tsa.plot_acf(residual, lags=40)
    plt.show()
    return res

def heteroscedasticity(dataframe):
    columns = ['Unnamed: 0', 'year', 'popularity', 'name', 'artists', 'top genre']
    data = dataframe.drop(columns, 1)
    for i in data.columns.tolist():
        F, p = stats.f_oneway(data[i])
        if p < 0.05:
            print("reject null hypothesis")    # null hypothesis : heteroscedasticity isn`t included
        else:
            print("accept null hypothesis")
    return p

def pearson_coeff(dataframe):
    columns = ['name','artists','top genre','Unnamed: 0','year','valence','acousticness','speechiness','liveness','popularity']
    data = dataframe.drop(columns, 1)
    dict_for_coeff = {}
    for i in data.columns.tolist():
        lst = data[i].to_numpy()
        lst1 = data['LOG_popularity'].to_numpy()
        pearson_coeff = pearsonr(lst, lst1)[0]
        if pearson_coeff > 0:
            dict_for_coeff[i] = pearson_coeff
    return dict_for_coeff


d = pd.read_csv('C:\\Users\Влада Гейфман\Desktop\python\course_work\\top.csv',encoding='latin-1')
d = d.rename(columns = {'val':'valence','dur':'duration_ms','acous':'acousticness','spch':'speechiness','pop':'popularity'})
d = d.rename(columns = {'title':'name','artist':'artists','bpm':'tempo','nrgy':'energy','dnce':'danceability','dB':'loudness','live':'liveness'})
data = d.drop_duplicates()
data['LOG_popularity'] = np.log(data['popularity']).dropna().replace(-np.inf, -0.00000001)
data = detecting_outliers(data)
prepare_data = ['valence','acousticness','speechiness','liveness']
for i in prepare_data:
    lst = data[i].to_numpy()
    label = i+'01'
    data[label] = np.divide(lst,100)
correlation_coeff = pearson_coeff(data)
feat_for_model = data[['tempo','danceability','acousticness01','speechiness01']]
target_col = data['LOG_popularity']
regr = linear_model.LinearRegression()
regr.fit(feat_for_model,target_col)
path  = input('Path to your song:')            
path1 = input('Path to your text of song:')
song,rate = lb.load(path,sr=None, mono=True,  dtype=np.float32)
features = []
file = open(path1, "rt")
data = file.read()
words = data.split()
count_words = len(words)
duration = lb.get_duration(song)
tempo,beats = lb.beat.beat_track(song)
speechiness = count_words/duration
accousticess = 1-speechiness
danceability = tempo/100
features.append(tempo)
features.append(danceability)
features.append(accousticess)
features.append(speechiness)
s = np.array(features)
prediction = regr.predict(np.reshape(s,(1,-1)))
res = str(np.exp(prediction))
print('Predicted popularity:',res[1:len(res)-1])