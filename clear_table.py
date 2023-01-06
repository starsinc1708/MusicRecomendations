import pandas as pd


df = pd.read_csv('Data Collection/all_spotify_tracks.csv', sep=",")
df = df[df['id'].str.contains('"') == False]
df.drop_duplicates(subset='id', inplace=True)
df.to_csv('Data Collection/all_spotify_tracks.csv', index=False)
