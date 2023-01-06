import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

playlist_url = 'https://open.spotify.com/playlist/51WDRItSGTxLXWLLzyW08g'


if __name__ == "__main__":
    client_id = "1325c0f83be24e988096fdf61f1f94d1"
    client_secret = "105cb7762cd64ace870270f6831f65b2"
    redirect_uri = "http://localhost:3000"
    playlist_url = playlist_url.split("?")[0]
    scope = "user-library-read"

    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)

    user_playlist_tracks = {'id': [],
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

    i = 0
    while True:
        results = sp.playlist_tracks(playlist_id=playlist_url,
                                     limit=50, offset=i)
        if len(results['items']) != 0:
            for idx, item in enumerate(results['items']):
                track = item['track']
                user_playlist_tracks['main_artist'].append(track['artists'][0]['name'])
                user_playlist_tracks['title'].append(track['name'])
                user_playlist_tracks['id'].append(track['id'])
            i += 50
        else:
            break

    print(f"---------------------------------\n"
          f"Audio Features collection started\n"
          f"PLAYLIST_ID = {playlist_url}\n"
          f"PLAYLIST_NAME = {sp.playlist(playlist_id=playlist_url)['name']}\n"
          f"---------------------------------")

    for i in range(len(user_playlist_tracks['title'])):
        tf = sp.audio_features(user_playlist_tracks['id'][i])
        user_playlist_tracks['acousticness'].append(tf[0]['acousticness'])
        user_playlist_tracks['danceability'].append(tf[0]['danceability'])
        user_playlist_tracks['energy'].append(tf[0]['energy'])
        user_playlist_tracks['duration_ms'].append(tf[0]['duration_ms'])
        user_playlist_tracks['instrumentalness'].append(tf[0]['instrumentalness'])
        user_playlist_tracks['key'].append(tf[0]['key'])
        user_playlist_tracks['liveness'].append(tf[0]['liveness'])
        user_playlist_tracks['loudness'].append(tf[0]['loudness'])
        user_playlist_tracks['mode'].append(tf[0]['mode'])
        user_playlist_tracks['speechiness'].append(tf[0]['speechiness'])
        user_playlist_tracks['tempo'].append(tf[0]['tempo'])
        user_playlist_tracks['time_signature'].append(tf[0]['time_signature'])
        user_playlist_tracks['valence'].append(tf[0]['valence'])

    print("Audio Features collection is finished\n"
          "---------------------------------")
    user_playlist_tracks['duration_min'] = list()

    for i in range(len(user_playlist_tracks['duration_ms'])):
        user_playlist_tracks['duration_min'].append(user_playlist_tracks['duration_ms'][i] / 60000)

    tracks_DataFrame = pd.DataFrame.from_dict(user_playlist_tracks)

    tracks_DataFrame.to_csv('/Data Collection/all_spotify_tracks.csv',
                            mode='a',
                            header=False,
                            index=False)

    df = pd.read_csv('/Data Collection/all_spotify_tracks.csv', sep=",")
    df = df[df['id'].str.contains('"') == False]
    df.drop_duplicates(subset='id', inplace=True)
    df.to_csv('/Data Collection/all_spotify_tracks.csv', index=False)
