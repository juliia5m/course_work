import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import course

#tab = pd.read_csv('data_Chris Standring.csv',delimiter=',',index_col = 'date',parse_dates=True)
#tab = tab.drop(['isPartial'],axis = 1)
#mean = tab.loc[tab['date'][1]:tab['date'][len(tab)]].mean()
#var = tab.loc['2015-Feb':'2018-Feb'].var()
#sample = tab.loc['2015-Feb':'2018-Feb']
#sample.plot()
#plt.show()

path = input('Введить путь к файлу: ')
all_data_authors = []
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
        additional = []                         #массив для хранения характеристик популярности автора до выхода песни
        year = str(k) + '-'+'12'+'-'+'01'       #в датасете есть только год выпуска песни,по-этому формируем полную дату
        d = tab[tab['date'] <= year]
        new_data = d[j[5:len(j)-4]]
        new_d = d['date']
        x1 = []
        for i in range (len(new_d)):
            x1.append(float(new_d[i][:4]))      #составляем массив в котором будет только год для линейной регрессии
        x = np.array(x1).reshape(-1,1)
        mean = new_data.mean()
        var= new_data.var()
        delta = new_data[len(new_data)-1] - new_data[1]
        #x = new_d.to_numpy().reshape(-1,1)
        y = new_data.to_numpy()
        model = LinearRegression().fit(x,y)
        b1 = model.coef_
        R = model.score(x,y)
        additional.append(mean)
        additional.append(var)
        additional.append(delta)
        additional.append(b1)
        additional.append(R)
        all_data_authors.append(additional)
print(all_data_authors)