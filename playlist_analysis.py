import spotipy
from matplotlib import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from spotipy import SpotifyClientCredentials

playlist_id = 'https://open.spotify.com/playlist/51WDRItSGTxLXWLLzyW08g'

if __name__ == "__main__":

    client_id = "1325c0f83be24e988096fdf61f1f94d1"
    client_secret = "105cb7762cd64ace870270f6831f65b2"
    redirect_uri = "http://localhost:3000"

    playlist_url = playlist_id.split("?")[0]
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
        results = sp.playlist_tracks(playlist_id=playlist_id,
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
          f"PLAYLIST_ID = {playlist_id}\n"
          f"PLAYLIST_NAME = {sp.playlist(playlist_id=playlist_id)['name']}\n"
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

    ax = sns.kdeplot(tracks_DataFrame['duration_min'], fill=True, color="#1DB954", cut=0)

    plt.xlabel("Длительность (в минутах)")
    plt.ylabel("Распределение (в %)")
    plt.title("Распределенность треков по длительности")
    plt.show()

    # Строим 9 графиков разом
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3, 3), dpi=200)
    fig.subplots_adjust(hspace=1.5, wspace=1)

    pylab.subplot(3, 3, 1)
    ac = sns.kdeplot(tracks_DataFrame['acousticness'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 2)
    ac = sns.kdeplot(tracks_DataFrame['danceability'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 3)
    ac = sns.kdeplot(tracks_DataFrame['instrumentalness'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 4)
    ac = sns.kdeplot(tracks_DataFrame['liveness'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 5)
    ac = sns.kdeplot(tracks_DataFrame['loudness'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 6)
    ac = sns.kdeplot(tracks_DataFrame['speechiness'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 7)
    ac = sns.kdeplot(tracks_DataFrame['tempo'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 8)
    ac = sns.kdeplot(tracks_DataFrame['valence'], fill=True, color="green", cut=0)
    pylab.subplot(3, 3, 9)
    ac = sns.kdeplot(tracks_DataFrame['energy'], fill=True, color="green", cut=0)

    plt.show()