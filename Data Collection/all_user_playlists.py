import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

user_uri = 'oha2uhj671pwaswod0ytcvbjt'

if __name__ == "__main__":
    client_id = "1325c0f83be24e988096fdf61f1f94d1"
    client_secret = "105cb7762cd64ace870270f6831f65b2"
    redirect_uri = "http://localhost:3000"
    scope = "user-library-read"

    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)

    playlists = sp.user_playlists(user=user_uri)
    j = 0
    count_of_playlists = 0
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            count_of_playlists += 1
            playlist_tracks = {'id': [],
                               'playlist_id': [],
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

            # получаем все треки текущего плейлиста
            while True:
                tracks = sp.playlist_tracks(playlist_id=playlist['uri'].split(':')[2],
                                            limit=50,
                                            offset=j)
                if len(tracks['items']) != 0:
                    # проходим по всем трекам из заполняем артиста, название, id плейлиста, id трека
                    for idx, item in enumerate(tracks['items']):
                        track = item['track']
                        playlist_tracks['main_artist'].append(track['artists'][0]['name'])
                        playlist_tracks['title'].append(track['name'])
                        playlist_tracks['playlist_id'].append(playlist['uri'].split(':')[2])
                        playlist_tracks['id'].append(track['id'])
                    j += 50
                else:
                    break

            j = 0

            print(f"---------------------------------\n"
                  f"Audio Features collection started\n"
                  f"PLAYLIST_ID = {playlist['uri'].split(':')[2]}\n"
                  f"NAME = {playlist['name']}\n"
                  f"---------------------------------")

            # заполняем audio features текущего плейлиста
            for k in range(len(playlist_tracks['title'])):
                tf = sp.audio_features(playlist_tracks['id'][k])
                playlist_tracks['acousticness'].append(tf[0]['acousticness'])
                playlist_tracks['danceability'].append(tf[0]['danceability'])
                playlist_tracks['energy'].append(tf[0]['energy'])
                playlist_tracks['duration_ms'].append(tf[0]['duration_ms'])
                playlist_tracks['instrumentalness'].append(tf[0]['instrumentalness'])
                playlist_tracks['key'].append(tf[0]['key'])
                playlist_tracks['liveness'].append(tf[0]['liveness'])
                playlist_tracks['loudness'].append(tf[0]['loudness'])
                playlist_tracks['mode'].append(tf[0]['mode'])
                playlist_tracks['speechiness'].append(tf[0]['speechiness'])
                playlist_tracks['tempo'].append(tf[0]['tempo'])
                playlist_tracks['time_signature'].append(tf[0]['time_signature'])
                playlist_tracks['valence'].append(tf[0]['valence'])

            print("Audio Features collection is finished\n")
            playlist_tracks['duration_min'] = list()

            for l in range(len(playlist_tracks['duration_ms'])):
                playlist_tracks['duration_min'].append(playlist_tracks['duration_ms'][l] / 60000)

            tracks_DataFrame = pd.DataFrame.from_dict(playlist_tracks)

            tracks_DataFrame.to_csv('all_spotify_tracks.csv',
                                    mode='a',
                                    header=False,
                                    index=False)

            df = pd.read_csv('all_spotify_tracks.csv', sep=",")
            df = df.drop(np.where(df['title'] == '')[0])
            df.drop_duplicates(subset='id', inplace=True)
            # Write the results to a different file
            df.to_csv('all_spotify_tracks.csv', index=False)

        # если можно переходим к следующему плейлисту
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None

    print(f"Downloaded from {count_of_playlists} playlists")
