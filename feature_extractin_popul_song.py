import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import course

def compute_mean(dataframe):
    result = dataframe.mean()
    return result

def compute_var(dataframe):
    result = dataframe.var()
    return result

def compute_delta(dataframe1):
    # dataframe1 - pandas series где храняться только значения в определенную дату
    s = dataframe1.index[0]
    s1 = dataframe1.index[len(dataframe1)-1]
    result = dataframe1[s1] - dataframe1[s]
    return result

def compute_coeff_of_linear_regression(dataframe,dataframe1):
    # dataframe - pandas series где храняться только даты
    # dataframe1 - pandas series где храняться только значения в определенную дату
    x = []
    for i in range(dataframe.index[0], dataframe.index[len(dataframe) - 1] + 1):
        x.append(float(dataframe[i][:4]))
    arr = np.array(x).reshape(-1,1)
    arr1 = dataframe1.to_numpy()
    model = LinearRegression().fit(arr, arr1)
    b1 = model.coef_
    r = model.score(arr, arr1)
    return float(b1),r

def visualization_of_kmeans(arr):
    data_a = np.array(arr)
    pred  = KMeans(n_clusters=5).fit_predict(data_a)
    plt.scatter(data_a[:, 0], data_a[:, 1], c=pred)
    plt.show()
    plt.savefig('k-means.png')


all_data_songs = []
path = 'C:\\Users\Влада Гейфман\PycharmProjects\course_work\data_songs'
data = course.d_s_y
list_song = course.list_of_songs
for j in list_song:
    print(j)
    y = 0
    tab = pd.read_csv(path +'\\'+'data_' + j + '.csv',delimiter=',',parse_dates=True)
    for col in data:
        for i in range (len(data)):
            if j == data[col][i]:
                y = data['year'][i]
    if y != 2010:                                 # не хочет работать с песнями 2010 года
        additional = []
        year = str(y) + '01' + '01'
        interested_data = tab[tab['date'] >= year]
        dates = interested_data['date']
        values = interested_data[j]
        print(compute_mean(values))
        additional.append(compute_mean(values))
        additional.append(compute_var(values))
        additional.append(compute_delta(values))
        additional.append(compute_coeff_of_linear_regression(dates,values)[0])
        additional.append(compute_coeff_of_linear_regression(dates, values)[1])
        all_data_songs.append(additional)

saving = pd.DataFrame(all_data_songs).to_csv('data_song.csv')
visualization_of_kmeans(all_data_songs)
