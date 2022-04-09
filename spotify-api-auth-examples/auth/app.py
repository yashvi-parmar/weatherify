''' Example of Spotify authorization code flow (refreshable user auth).

Displays profile information of authenticated user and access token
information that can be refreshed by clicking a button.

Basic flow:
    -> '/'
    -> Spotify login page
    -> '/callback'
    -> get tokens
    -> use tokens to access API

Required environment variables:
    FLASK_APP, CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SECRET_KEY

More info:
    https://developer.spotify.com/documentation/general/guides/authorization-guide/#authorization-code-flow

'''
import os
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import RandomForestRegressor 
import numpy as np
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report, confusion_matrix
import requests, json
from geopy.geocoders import Nominatim
import geocoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import pickle
from dotenv import load_dotenv
load_dotenv() 

from flask import (
    abort,
    Flask,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import json
import logging
import os
import requests
import secrets
import string
from urllib.parse import urlencode
import random


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG
)


# Client info
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')


# Spotify API endpoints
AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
ME_URL = 'https://api.spotify.com/v1/me'


# Start 'er up
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/<loginout>')
def login(loginout):
    '''Login or logout user.

    Note:
        Login and logout process are essentially the same. Logout forces
        re-login to appear, even if their token hasn't expired.
    '''

    # redirect_uri can be guessed, so let's generate
    # a random `state` string to prevent csrf forgery.
    state = ''.join(
        secrets.choice(string.ascii_uppercase + string.digits) for _ in range(16)
    )

    # Request authorization from user
    scope = 'user-read-private user-read-email playlist-modify-public user-top-read'

    if loginout == 'logout':
        payload = {
            'client_id': CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': REDIRECT_URI,
            'state': state,
            'scope': scope,
            'show_dialog': True,
        }
    elif loginout == 'login':
        payload = {
            'client_id': CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': REDIRECT_URI,
            'state': state,
            'scope': scope,
        }
    else:
        abort(404)

    res = make_response(redirect(f'{AUTH_URL}/?{urlencode(payload)}'))
    res.set_cookie('spotify_auth_state', state)

    return res


@app.route('/callback')
def callback():
    error = request.args.get('error')
    code = request.args.get('code')
    state = request.args.get('state')
    stored_state = request.cookies.get('spotify_auth_state')

    # Check state
    if state is None or state != stored_state:
        app.logger.error('Error message: %s', repr(error))
        app.logger.error('State mismatch: %s != %s', stored_state, state)
        abort(400)

    # Request tokens with code we obtained
    payload = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
    }

    # `auth=(CLIENT_ID, SECRET)` basically wraps an 'Authorization'
    # header with value:
    # b'Basic ' + b64encode((CLIENT_ID + ':' + SECRET).encode())
    res = requests.post(TOKEN_URL, auth=(CLIENT_ID, CLIENT_SECRET), data=payload)
    res_data = res.json()

    if res_data.get('error') or res.status_code != 200:
        app.logger.error(
            'Failed to receive token: %s',
            res_data.get('error', 'No error information received.'),
        )
        abort(res.status_code)

    # Load tokens into session
    session['tokens'] = {
        'access_token': res_data.get('access_token'),
        'refresh_token': res_data.get('refresh_token'),
    }

    return redirect(url_for('home'))


@app.route('/refresh')
def refresh():
    '''Refresh access token.'''

    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': session.get('tokens').get('refresh_token'),
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    res = requests.post(
        TOKEN_URL, auth=(CLIENT_ID, CLIENT_SECRET), data=payload, headers=headers
    )
    res_data = res.json()

    # Load new token into session
    session['tokens']['access_token'] = res_data.get('access_token')

    return json.dumps(session['tokens'])


@app.route('/me')
def me():
    '''Get profile info as a API example.'''

    # Check for tokens
    if 'tokens' not in session:
        app.logger.error('No tokens in session.')
        abort(400)

    # Get profile info
    headers = {'Authorization': f"Bearer {session['tokens'].get('access_token')}"}

    res = requests.get(ME_URL, headers=headers)
    res_data = res.json()
    
    if res.status_code != 200:
        app.logger.error(
            'Failed to get profile info: %s',
            res_data.get('error', 'No error message returned.'),
        )
        abort(res.status_code)

        print(res_data)

    return render_template('me.html', data=res_data, tokens=session.get('tokens'))

#Top 100 of the user's songs
def _query_top():
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {session['tokens'].get('access_token')}"
    }

    params = {
        'time_range': 'medium_term',
        'limit': '50',
        'offset': '0',
    }

    response = requests.get('https://api.spotify.com/v1/me/top/tracks', headers=headers, params=params)
   
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {session['tokens'].get('access_token')}"
    }

    params = {
        'time_range': 'medium_term',
        'limit': '50',
        'offset': '49',
    }

    response2 =  requests.get('https://api.spotify.com/v1/me/top/tracks', headers=headers, params=params)

    if response.status_code == 200 and response2.status_code == 200:
        res_data = response.json()
        d = json.dumps(res_data)
        dict = json.loads(d)

        res_data2 = response2.json()
        d2 = json.dumps(res_data2)
        dict2 = json.loads(d2)

        z = {**dict, **dict2}
        return z
    else:
        return "None"


#Get audio features
def _query_audio(id):
    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {session['tokens'].get('access_token')}",
    }
    return requests.get('https://api.spotify.com/v1/audio-features/' + id, headers=headers)

#Get Location
def _find_location():
    g = geocoder.ip('me')
    Latitude = str(g.latlng[0])
    Longitude = str(g.latlng[1])

    # initialize Nominatim API
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(Latitude+","+Longitude)
    address = location.raw['address']
        
    # traverse the data
    city = address.get('city', '')
    #Open Weather API
    api_key = "767a9a6ec8856b1d5f4e998eb195f561"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
        
    # Give city name
    city_name = city
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]
    
        # store the value corresponding
        # to the "pressure" key of y
        current_pressure = y["pressure"]
        
        # store the value corresponding
        # to the "humidity" key of y
        current_humidity = y["humidity"]
        
        # store the value of "weather"
        # key in variable z
        z = x["weather"]
        
        # store the value corresponding
        # to the "description" key at
        # the 0th index of z
        weather_description = z[0]["main"]
        description = z[0]["description"]
        
        
    else:
        print(" City Not Found ")
        current_temperature = 10
        city_name = 'Toronto'
        description = 'Broken Clouds'
        weather_description = 'Clouds'

    dict = {'Temperature': current_temperature,
            'Location': city_name,
            'Description': description, 
            'Main': weather_description}
    return dict

def _query_user():
    headers = {'Authorization': f"Bearer {session['tokens'].get('access_token')}"}

    return requests.get(ME_URL, headers=headers)

def _query_recommend(seedtracks, energy1, energy2, valence1, valence2):
    headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {session['tokens'].get('access_token')}",
        }

    params = {
            'limit': '25',
            'seed_tracks': seedtracks,
            'min_energy': energy2,
            'max_energy': energy1,
            'min_valence': valence2,
            'max_valence': valence1
        
        }

    return requests.get('https://api.spotify.com/v1/recommendations', headers=headers, params=params)

#Get all of the user's playlist (hardcoded 50)   
def _query_playlist(user):
    headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {session['tokens'].get('access_token')}",
        }

    params = {
            'limit': '50',
            'offset': '0',
        }

    return requests.get('https://api.spotify.com/v1/users/' + user +'/playlists', headers=headers, params=params)

@app.route('/home', methods=['GET'])
def home():
    dict = _query_top()
    if dict != None:
    
        songs = dict["items"]
        ids = [] 
        audiofeatures = []
        dict2 = dict
        #Find the songs id and audio features
        for i in range(len(songs)): 
            ids.append(songs[i]["id"])
        #Find the audio features of the songs
        for k in range(len(ids)):
            audio = _query_audio(ids[k])
            res_data = audio.json()
            d = json.dumps(res_data)
            dict = json.loads(d)
            audiofeatures.append(dict)

        df = pd.DataFrame(audiofeatures)
        df = df.drop_duplicates()
        #df.to_csv('datausers.csv')   

        # load dataset of training data
        data = pd.read_csv("data.csv")

    #split dataset in features and target variable
        feature_cols = ['energy','valence']

        X = data[feature_cols] # Features
        y = data.weather # Target variable

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 70% training and 30% test

        #Split the user's data
        data2 = df
        X2 = data2[feature_cols]

        clf = RandomForestClassifier(n_estimators= 160, min_samples_split= 5, min_samples_leaf= 1, max_features= 'auto', max_depth= 10, bootstrap= True)
        clf.fit(X_train, y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        
        #Confusion Matrix
        #cm= confusion_matrix(y_test, y_pred)  
        #print(cm)

        #Get user's location and weather
        dict3 = _find_location()
        city_name = dict3.get('Location')
        temperature = dict3.get('Temperature')
        weather = dict3.get('Weather')
        description = dict3.get('Description')
        weather_description = dict3.get('Main')
        
        with open('model_pkl', 'wb') as files:
            pickle.dump(clf, files)
        pickled_model = pickle.load(open('model_pkl', 'rb'))

        data2['weather'] = pickled_model.predict(X2)
        
        if weather_description == 'Thunderstorm':
            a = 3
        elif weather_description == "Drizzle":
            a = 3
        elif weather_description == "Rain":
            a = 3
        elif weather_description == "Snow":
            a = 3
        elif weather_description == "Mist":
            a = 2
        elif weather_description == "Fog":
            a = 2
        elif weather_description == "Clear":
            a = 1
        elif weather_description == "Clouds":
            a = 2

        df = data2.drop(data2[data2.weather != a].index)
        df = df.drop(columns=['weather'])

        energy1 = df["energy"].max()
        valence1 = df["valence"].max()

        energy2 = df["energy"].min()
        valence2 = df["valence"].min()

        #Take a random sample from the user's top 100 
        newdf = pd.DataFrame()
        newdf = df.sample(n= 4)
        seedtracks = newdf['id']

        track = []
        artist = []

        for i in range(4):
            track.append(newdf['id'].iloc[i])
            res3 = _query_user()
            res_data3 = res3.json()
            d3 = json.dumps(res_data3)
            dict3 = json.loads(d3) 
            user = dict3['id']
        
        #Using the recommendations api, recommending songs based on sample tracks
        recommend = _query_recommend(seedtracks, energy1, energy2, valence1, valence2)
        res_data4 = recommend.json()
        d4 = json.dumps(res_data4)
        dict4 = json.loads(d4)
        #Find all the songs and their audio features
        songs = dict4['tracks']
        list2 = []

        ids = [] 
        artist = []
        track = []
            
        audiofeatures = []

        #Find the songs id and audio features
        for i in range(len(songs)): 
            ids.append(songs[i]["id"])
        #Find audio features    
        for k in range(len(ids)):
            audio = _query_audio(ids[k])
            res_data2 = audio.json()
            d2 = json.dumps(res_data2)
            dict2 = json.loads(d2)
            audiofeatures.append(dict2)

        newdf2 = pd.DataFrame(audiofeatures)

        result = newdf2
        #result.to_csv('data2.csv')

        items = []
        for i in range(len(result)):
            items.append(result['uri'].iloc[i])
            
        name = 'Weather Playlist'
        dict_playlist = _query_playlist(user)    
        res_data = dict_playlist.json()
        d = json.dumps(res_data)
        play = json.loads(d)  
        a = 0

        length = play['total']
        print(length)
        #Find if the playlist already exists
        for i in range(length):
            item = play["items"]
            if (item != None):
                if (item[i]["name"] == name):
                    print(name)
                    a = 1
                    
        #If the playlist does not exist            
        if a == 0:
            headers = {
                    'Accept': 'application/json',
                    # Already added when you pass json=
                    # 'Content-Type': 'application/json',
                    'Authorization': f"Bearer {session['tokens'].get('access_token')}",
            }

            json_data = {
                    'name': 'Weather Playlist',
                    'public': True,
            }

            playlist = requests.post('https://api.spotify.com/v1/users/' + user +'/playlists', headers=headers, json=json_data)
                
            playlistdata = playlist.json()
            p = json.dumps(playlistdata)
            playlist = json.loads(p)
            playlist_id = playlist['id']
            print(playlist_id)

            for i in range(15):

                    headers = {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json',
                        'Authorization': f"Bearer {session['tokens'].get('access_token')}",
                    }

                    params = {
                        'uris': items[i],
                    }

                    response = requests.post('https://api.spotify.com/v1/playlists/'+ playlist_id + '/tracks', headers=headers, params=params)
                    print(items)

            data = {
            'Location': city_name,
            'Temperature': temperature,
            'Description': weather_description,
            }
            json_object = json.dumps(data, indent = 4) 
        temp = round(float(temperature) - 273, 0)

        return render_template('home.html', Location=city_name, Temperature=temp,  Description=description, tokens=session.get('tokens'))
    
    else:
        return render_template('index.html')

    
