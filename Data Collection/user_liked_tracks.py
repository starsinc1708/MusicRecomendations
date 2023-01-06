import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

if __name__ == "__main__":

    client_id = "1325c0f83be24e988096fdf61f1f94d1"
    client_secret = "105cb7762cd64ace870270f6831f65b2"
    redirect_uri = "http://localhost:3000"
    username = "oha2uhj671pwaswod0ytcvbjt"

    scope = "user-library-read"

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(client_id=client_id,
                                  client_secret=client_secret,
                                  redirect_uri=redirect_uri,
                                  username=username,
                                  scope=scope))

    lib_of_tracks = {'id': [],
                     'title': [],
                     'main_artist': [],
                     'acousticness': [],
                     'danceability': [],
                     'energy': [],
                     'duration_ms': [],
                     'instrumentalness': [],
                     'key': [],
                     'liveness': [],
                     'loudness': [],
                     'mode': [],
                     'speechiness': [],
                     'tempo': [],
                     'time_signature': [],
                     'valence': []}

    print(f"USER_ID = {username}")

    # Получаем список треков
    # Записываем в словарь исполнителя, название трека и его id

    i = 0
    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=i)
        if len(results['items']) != 0:
            for idx, item in enumerate(results['items']):
                track = item['track']
                lib_of_tracks['main_artist'].append(track['artists'][0]['name'])
                lib_of_tracks['title'].append(track['name'])
                lib_of_tracks['id'].append(track['id'])
            i += 50
        else:
            break

    print("---------------------------------\n"
          "Audio Features collection started\n"
          "---------------------------------")

    for i in range(len(lib_of_tracks['title'])):
        tf = sp.audio_features(lib_of_tracks['id'][i])
        lib_of_tracks['acousticness'].append(tf[0]['acousticness'])
        lib_of_tracks['danceability'].append(tf[0]['danceability'])
        lib_of_tracks['energy'].append(tf[0]['energy'])
        lib_of_tracks['duration_ms'].append(tf[0]['duration_ms'])
        lib_of_tracks['instrumentalness'].append(tf[0]['instrumentalness'])
        lib_of_tracks['key'].append(tf[0]['key'])
        lib_of_tracks['liveness'].append(tf[0]['liveness'])
        lib_of_tracks['loudness'].append(tf[0]['loudness'])
        lib_of_tracks['mode'].append(tf[0]['mode'])
        lib_of_tracks['speechiness'].append(tf[0]['speechiness'])
        lib_of_tracks['tempo'].append(tf[0]['tempo'])
        lib_of_tracks['time_signature'].append(tf[0]['time_signature'])
        lib_of_tracks['valence'].append(tf[0]['valence'])

    print("Audio Features collection is finished\n"
          "---------------------------------")

    lib_of_tracks['duration_min'] = list()

    for i in range(len(lib_of_tracks['duration_ms'])):
        lib_of_tracks['duration_min'].append(lib_of_tracks['duration_ms'][i] / 60000)

    tracks_DataFrame = pd.DataFrame.from_dict(lib_of_tracks)
    tracks_DataFrame.to_csv('your_liked_tracks.csv',
                            index=False)
