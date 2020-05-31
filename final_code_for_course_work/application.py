from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.lang import Builder

import os
import pandas as pd
import numpy as np
from scipy import stats
import librosa as lb
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

d = pd.read_csv('C:\\Users\\Julia\PycharmProjects\kiwi_project\\top.csv',encoding='latin-1')
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


feat_for_model = data[['tempo','danceability','acousticness01','speechiness01']]
target_col = data['LOG_popularity']
regr = linear_model.LinearRegression()
regr.fit(feat_for_model,target_col)
def get_features(path, path1):
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
    return 'PREDICTED POPULARITY:', res[1: len(res) - 1]

def features_for_user_file(path, path1):
    song, rate = lb.load(path, sr=None, mono=True, dtype=np.float32)
    features = []

    file = open(path1, "rt")
    data = file.read()
    words = data.split()
    count_words = len(words)
    duration = lb.get_duration(song)
    tempo, beats = lb.beat.beat_track(song)
    speechiness = count_words / duration
    accousticess = 1 - speechiness
    danceability = tempo / 100
    features.append(tempo)
    features.append(danceability)
    features.append(accousticess)
    features.append(speechiness)
    s = np.array(features)
    prediction = regr.predict(np.reshape(s, (1, -1)))
    res = str(np.exp(prediction))
    a = 'PREDICTED POPULARITY:', res[1: len(res) - 1]
    sp = 'counted speechiness:', speechiness
    ac = 'counted accousticess', accousticess
    dc = 'counted danceability', danceability


    listt = [sp, ac, dc, a ]
    return listt







Builder.load_file('application.kv')


class LoadSongDialog(AnchorLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class LoadTextDialog(AnchorLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(AnchorLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(AnchorLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._song = None
        self._text = None

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load_song(self):
        content = LoadSongDialog(load=self.load_song, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load song", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load_text(self):
        content = LoadTextDialog(load=self.load_text, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load text", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load_song(self, path, filename):
        print(path, filename)
        self._song = filename[0]

    def load_text(self, path, filename):
        print(path, filename)
        self._text = filename[0]





    def save(self, path, filename):
         file = os.path.join(path, filename)
         with open(file,'w') as temp:
             temp.write(str(features_for_user_file(self._song,self._text)))


    def result(self):
        self.label_res.text = str(get_features(self._song,self._text))


class PopularityPrediction(App):

    def build(self):
        return Root()


Factory.register('Root', cls=Root)
Factory.register('LoadSongDialog', cls=LoadSongDialog)
Factory.register('LoadTextDialog', cls=LoadTextDialog)
Factory.register('SaveDialog', cls=SaveDialog)


if __name__ == '__main__':
    PopularityPrediction().run()