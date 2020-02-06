import pandas as pd
import os

data = pd.read_csv('C:\\Users\Влада Гейфман\Desktop\songs.csv',delimiter=',')
data = data.drop(data[(data.year < 2004)].index)
new_file = data.to_csv('song1.csv')

data = pd.read_csv('C:\\Users\Влада Гейфман\PycharmProjects\course_work\song1.csv',delimiter=',')
col = data.columns.tolist()
d_a_s = data[[col[7],col[2]]]
d_a_s_y = data[[col[7],col[2],col[11]]]
d_s_y = data[[col[2],col[11]]]

path = input('Путь к папке с авторами:')
path1 = input('Путь к папке с песнями:')
list_of_csv_author = []
list_of_csv_songs = []
for top, dirs, files in os.walk(path):
    for nm in files:
        list_of_csv_author.append(nm)

for top, dirs, files in os.walk(path1):
    for nm in files:
        list_of_csv_songs.append(nm)

lst = []
for i in list_of_csv_songs:
    lst.append(i[5:len(i)-4])

all_songs = []
for j in list_of_csv_author:
    tab = pd.read_csv(path +'\\'+j,delimiter=',',parse_dates=True)
    for col in d_a_s_y:
        for i in range (len(d_a_s_y)):
            if j[5:len(j)-4] == d_a_s_y[col][i]:
                all_songs.append(d_a_s_y['title'][i])

list_of_songs = list(set(lst) & set(all_songs))
print(list_of_songs)



