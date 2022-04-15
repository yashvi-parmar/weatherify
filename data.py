import pandas as pd 
import spotipy 
sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 
import numpy as np

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 

list = ['37i9dQZF1DX0BcQWzuB7ZO', '37i9dQZF1DWTUHzPOW6Jl7', '37i9dQZF1DX2UgsUIg75Vg', '37i9dQZF1DX6VdMW310YC7', '37i9dQZF1DWVrtsSlLKzro', '37i9dQZF1DWSqBruwoIXkA']
# Energetic, Calm, Sad
list2 = [1, 1, 2, 2, 3, 3]

df = pd.DataFrame()
df3 = pd.DataFrame()
df["weather"] = np.nan
df2 = []
# Initiate a dictionairy with all the information you want to crawl
data_dict = {"id":[], "track_name":[], "artist_name":[],
             "valence":[], "energy":[]}

for x in range(len(list)):
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
    sp.trace=False 
    playlist = sp.user_playlist("yash", list[x]) 
    songs = playlist["tracks"]["items"] 
    ids = [] 
    for i in range(len(songs)): 
        ids.append(songs[i]["track"]["id"])     
    features = sp.audio_features(ids) 
    
    df = pd.DataFrame(features)
    df['weather'] = list2[x]  
    df2.append(df)


df3 = pd.concat(df2)
df3= df3.drop(columns=['danceability','key','loudness','mode','speechiness', 'acousticness',	'instrumentalness',	'liveness','tempo',	'type', 'track_href','analysis_url','duration_ms', 'time_signature', 'id', 'uri'])
df3.to_csv('data.csv')