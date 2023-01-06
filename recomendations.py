import spotipy
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
import warnings


warnings.filterwarnings('ignore')
playlist_url = 'https://open.spotify.com/playlist/51WDRItSGTxLXWLLzyW08g'


# преобразует треки из плейлиста в pandas.DataFrame
def create_data_from_playlist(uri, sp):
    playlist_id = uri.split('/')[4]

    results = sp.playlist_tracks(playlist_id=playlist_id,
                                 limit=1, offset=0)

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

    j = 0
    # получаем все треки текущего плейлиста
    while True:
        tracks = sp.playlist_tracks(playlist_id=playlist_id,
                                    limit=50,
                                    offset=j)
        if len(tracks['items']) != 0:
            # проходим по всем трекам из заполняем артиста, название, id плейлиста, id трека
            for idx, item in enumerate(tracks['items']):
                track = item['track']
                playlist_tracks['main_artist'].append(track['artists'][0]['name'])
                playlist_tracks['title'].append(track['name'])
                playlist_tracks['playlist_id'].append(playlist_id)
                playlist_tracks['id'].append(track['id'])
            j += 50
        else:
            break

    print(f"---------------------------------\n"
          f"Audio Features collection started\n"
          f"PLAYLIST_ID = {playlist_id}\n"
          f"PLAYLIST_NAME = {sp.playlist(playlist_id=playlist_id)['name']}\n"
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

    return playlist_tracks


# теоретическая часть (Анализ алгоритмов)
def algorithm_analysis(url):

    url = url.split("?")[0]
    df = create_data_from_playlist(url, sp)
    data = pd.DataFrame.from_dict(df)

    print(data.head())
    data = data.dropna()
    print(data.dtypes)
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_ms', 'time_signature']
    # DATA PREPROCESSING: Feature Transformation
    print("-------------------- DATA PREPROCESSING: Feature Transformation --------------------")
    '''
        I performed feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features.
    
        Given that HAC deals with ‘distances’ — abstract or otherwise — we need to standard-scale our data before feeding it into the clustering algorithm.
    
        This ensures that our final results are not skewed by feature units.
        For example, tempo typically ranges between 70 and 180 beats per minute, whereas most other measures fall somewhere between 0 and 1.
        Without scaling, two songs with very different tempos would always be very ‘far apart’, even if they were identical on the other metrics.
        '''
    scaler = StandardScaler()
    scaler.fit(data[features])
    features_scaled = scaler.transform(data[features])
    ########################################################################################################################################################
    print("------------------------ PCA: Principal Component Analysis ------------------------")
    '''
        PCA: Principal Component Analysis
        Principal Component Analysis (PCA) is one of the most common linear dimensionality reduction techniques. 
        It emphasizes variation and brings out strong patterns in a dataset. 
        In other words, it takes all the variables then represents it in a smaller space while keeping the nature of the original data as much as possible.
    
        The first principal component will encompass as much of the dataset variation as possible in 1 dimension,
        The second component will encompass as much as possible of the remaining variation as possible while remaining orthogonal to the first, and so on
        '''
    # PCA with 3 components: 3D visualization
    pca = PCA(n_components=3, random_state=42)
    df_pca = pd.DataFrame(data=pca.fit_transform(features_scaled), columns=['PC1', 'PC2', 'PC3'])
    pca.components_
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1, 2], ["First component", "Second component", "Third component"])
    plt.colorbar()
    plt.xticks(range(len(data[features].columns)), data[features], rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()
    df_pca_fix = df_pca.merge(data, left_index=True, right_index=True)
    df_pca_fix = df_pca_fix[['PC1', 'PC2', 'PC3', 'title', 'main_artist', 'danceability']]
    pd.set_option('display.max_columns', None)
    print(df_pca_fix.head())
    pd.reset_option('display.max_columns')
    # Plot the PCA
    px.scatter_3d(df_pca_fix,
                  x='PC1',
                  y='PC2',
                  z='PC3',
                  title='Principal Component Analysis Projection (3-D)',
                  color='danceability',
                  size=np.ones(len(df_pca_fix)),
                  size_max=5,
                  height=600,
                  hover_name='title',
                  hover_data=['main_artist'],
                  color_continuous_scale=px.colors.cyclical.mygbm[:-6]).show()
    '''
        We can see each song position and its distance to other songs based on the audio features that have been transformed.
        Most points are concentrated on the green areas.
    
        The mapping also confirms that danceability does correlate with PC2 to some extent.
        '''
    # Dimensionality Reduction with n number of components ( Уменьшение размерности с n числом компонентов )
    pca = PCA()
    df_pca = pca.fit_transform(features_scaled)
    # Investigate the variance accounted for by each principal component.
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    plt.show()
    # Re-apply PCA to the data while selecting for number of components to retain.
    pca = PCA(n_components=10)
    df_pca = pca.fit_transform(features_scaled)
    pca.components_
    print(pca.explained_variance_ratio_)
    ########################################################################################################################################################
    print("------------------------ Clustering Techniques ------------------------")
    '''
        Clustering Techniques
        Clustering is the task of partitioning the dataset into groups of similarity called clusters
    
        Clustering algorithms assign a number to each datapoint indicating which cluster it belongs to.
    
        Data Clustering with K-Means
        The main idea behind k-means clustering is that we choose how many clusters we would like to create (typically we call that number k).
        We choose this based on domain knowledge (maybe we have some market research on the number of different types of groups we expect to see in our customers?), 
        based on a 'best-guess', or randomly.
    
        In the end you are left with areas that identify in which a cluster a newly assigned point would be classified.
        '''
    kmeans = KMeans(n_clusters=2)
    model = kmeans.fit(features_scaled)
    data_2 = data.copy()
    data_2 = data_2[features]
    data_2['labels'] = model.labels_
    print(data_2['labels'].value_counts())
    pd.set_option('display.max_columns', None)
    print(data_2.groupby('labels').mean())
    pd.reset_option('display.max_columns')
    sns.lmplot(data=data_2, x='valence', y='danceability', hue='labels', fit_reg=False, legend=True)
    plt.show()
    sns.lmplot(data=data_2, x='energy', y='loudness', hue='labels', fit_reg=False, legend=True)
    plt.show()
    sns.lmplot(data=data_2, x='danceability', y='loudness', hue='labels', fit_reg=False, legend=True)
    plt.show()
    sns.lmplot(data=data_2, x='danceability', y='acousticness', hue='labels', fit_reg=False, legend=True)
    plt.show()
    sns.lmplot(data=data_2, x='danceability', y='tempo', hue='labels', fit_reg=False, legend=True)
    plt.show()
    ########################################################################################################################################################
    print("------------------------ Number of Clusters = 9 ------------------------")
    '''
        Optimal Number of Clusters
        One of the hardest and most important parameters to optimize is the number of clusters.
    
        Having too many clusters might mean that we haven't actually learned much about the data - the whole point of clustering
        is to identify a relatively small number of similarities that exist in the dataset.
    
        Too few clusters might mean that we are grouping unlike samples together artificially.
    
        There are many different methods for choosing the appropriate number of clusters, 
        but one common method is calculating a metric for each number of clusters, 
        then plotting the error function vs the number of clusters.
    
        Yellowbrick's KElbowVisualizer:
        Yellowbrick's KElbowVisualizer implements the “elbow” method of selecting the optimal number of clusters
        by fitting the K-Means model with a range of values for K.
        '''
    X = features_scaled
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 10))
    visualizer.fit(X)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure
    # fits the model for a range of  K  values from 1 to 9, which is set by the parameter k=(1,10).
    # we see that the model is fit with 3 clusters - we can see an "elbow" in the graph.
    kmeans = KMeans(n_clusters=9)
    model = kmeans.fit(features_scaled)
    data_3 = data.copy()
    data_3['labels'] = model.labels_

    print(data_3['labels'].value_counts())
    pd.set_option('display.max_columns', None)
    print(data_3.groupby('labels').mean())
    pd.reset_option('display.max_columns')


# создания списка рекомендаций по плейлисту
def create_recomendations(url, num_of_recomendations=10):
    df = create_data_from_playlist(url, sp)
    data = pd.DataFrame.from_dict(df)

    features = ['danceability',
                'energy',
                'key',
                'loudness',
                'mode',
                'acousticness',
                'instrumentalness',
                'liveness',
                'valence',
                'tempo']
    all_songs = pd.read_csv('Data Collection/all_spotify_tracks.csv')
    # drop duplicates
    all_songs.drop_duplicates(subset="id", keep=False, inplace=True)
    # check recommended songs that are not in the playlist already:
    all_songs = all_songs.loc[~(all_songs.id.isin(data['id'])), :]
    ########### НЕКОНТРОЛИРУЕМОЕ ОБУЧЕНИЕ ###########
    scaler = StandardScaler()
    scaler.fit(all_songs[features])
    all_songs_features_scaled = scaler.transform(all_songs[features])

    # PCA with all songs playlist (метод главных компонент)
    pca = PCA(n_components=3, random_state=42)
    df_pca_all_songs = pd.DataFrame(data=pca.fit_transform(all_songs_features_scaled), columns=['PC1', 'PC2', 'PC3'])
    df_pca_all_songs = df_pca_all_songs.merge(all_songs, left_index=True, right_index=True)
    df_pca_all_songs = df_pca_all_songs[['PC1', 'PC2', 'PC3', 'title', 'main_artist']]

    # PCA with entered playlist
    df_scaled = scaler.transform(data[features])

    # using trained PCA
    df_pca = pd.DataFrame(data=pca.transform(df_scaled), columns=['PC1', 'PC2', 'PC3'])
    df_pca = df_pca.merge(data, left_index=True, right_index=True)
    df_pca = df_pca[['PC1', 'PC2', 'PC3', 'title', 'main_artist']]

    ############### ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ ###############
    columns = ['PC1', 'PC2', 'PC3']
    kdB = KDTree(df_pca_all_songs[columns].values)  # all songs
    neighbours = kdB.query(df_pca[columns].values, k=1)[-1]
    recomendations = all_songs[all_songs.index.isin(neighbours[:num_of_recomendations])]
    recomendations_output = recomendations[['title', 'main_artist']]
    recomendations_output.columns = ['Song Recommendation', 'Artist']
    print(recomendations_output)


if __name__ == "__main__":
    client_id = "1325c0f83be24e988096fdf61f1f94d1"
    client_secret = "105cb7762cd64ace870270f6831f65b2"
    redirect_uri = "http://localhost:3000"
    playlist_url = playlist_url.split("?")[0]
    scope = "user-library-read"

    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)

    #create_recomendations(playlist_url, num_of_recomendations=20)

    algorithm_analysis(playlist_url)
