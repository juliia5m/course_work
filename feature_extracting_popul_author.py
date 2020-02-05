import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os
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


path = input('Введить путь к файлу: ')
all_data_authors = []                     #масив для дальнейшего анализа характеристик популярности автора до выхода песни
all_data_authors_after = []               #масив для дальнейшего анализа характеристик популярности автора после выхода песни
data = course.d_a_s_y
list_of_csv_author = []
for top, dirs, files in os.walk(path):
    for nm in files:
        list_of_csv_author.append(nm)

for j in list_of_csv_author:
    all_years = []
    tab = pd.read_csv(path +'\\'+j,delimiter=',',parse_dates=True)
    for col in data:
        for i in range (len(data)):
            if j[5:len(j)-4] == data[col][i]:
                all_years.append(data['year'][i])
    for k in all_years:
        additional = []                          #массив для хранения характеристик популярности автора до выхода песни
        additional_after = []                    #масив для хранения характеристик популярности автора после выхода песни
        year = str(k) + '-'+'01'+'-'+'01'       #в датасете есть только год выпуска песни,по-этому формируем полную дату
        d_before = tab[tab['date'] <= year]
        d_after = tab[tab['date'] >= year]
        new_data_after = d_after[j[5:len(j)-4]]
        new_d_after = d_after['date']
        new_data = d_before[j[5:len(j)-4]]
        new_d = d_before['date']
        additional_after.append(compute_mean(new_data_after))
        additional_after.append(compute_var(new_data_after))
        additional_after.append(compute_delta(new_data_after))
        additional_after.append(max(new_data_after))
        additional_after.append(compute_coeff_of_linear_regression(new_d_after,new_data_after)[0])
        additional_after.append(compute_coeff_of_linear_regression(new_d_after, new_data_after)[1])
        all_data_authors_after.append(additional_after)
        additional.append(compute_mean(new_data))
        additional.append(compute_var(new_data))
        additional.append(max(new_data))
        additional.append(compute_delta(new_data))
        additional.append(compute_coeff_of_linear_regression(new_d,new_data)[0])
        additional.append(compute_coeff_of_linear_regression(new_d,new_data)[1])
        all_data_authors.append(additional)
datafr1 = pd.DataFrame(all_data_authors).to_csv('data1.csv')
datafr = pd.DataFrame(all_data_authors_after).to_csv('data1.csv')
visualization_of_kmeans(all_data_authors)