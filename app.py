from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import h5py
import os

app = Flask(__name__)

# ========================
# Utility to extract weights
# ========================
def extract_weights(file_path, layer_name):
    with h5py.File(file_path, 'r') as h5_file:
        if layer_name in h5_file:
            weights = h5_file[layer_name][()]
            weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
            return weights
    raise KeyError(f"Layer '{layer_name}' not found in HDF5 file.")


# ========================
# Load model and data
# ========================
file_path = 'model/myanimeweights.h5'
anime_weights = extract_weights(file_path, 'anime_embedding/anime_embedding/embeddings:0')

with open('model/anime_encoder.pkl', 'rb') as file:
    anime_encoder = pickle.load(file)

with open('model/anime-dataset-2023.pkl', 'rb') as file:
    df_anime = pickle.load(file)
df_anime = df_anime.replace("UNKNOWN", "")


# ========================
# Item-based Recommendation
# ========================
def find_similar_animes(anime_name, n=10):
    try:
        a_row = df_anime[df_anime['Name'] == anime_name].iloc[0]
        encoded_index = anime_encoder.transform([a_row['anime_id']])[0]
        dists = np.dot(anime_weights, anime_weights[encoded_index])
        sorted_idx = np.argsort(dists)[-n-1:]  # get top n similar
        similar_animes = []
        for i in sorted_idx:
            decoded_id = anime_encoder.inverse_transform([i])[0]
            anime = df_anime[df_anime['anime_id'] == decoded_id]
            if not anime.empty:
                anime = anime.iloc[0]
                similar_animes.append({
                    "anime_id": anime.anime_id,
                    "name": anime.Name,
                    "img": anime["Image URL"],
                    "genres": anime.Genres,
                    "score": anime.Score,
                    "synopsis": anime.Synopsis
                })
        return pd.DataFrame(similar_animes).sort_values(by="score", ascending=False)
    except Exception as e:
        print(f"[find_similar_animes error]: {e}")
        return pd.DataFrame()


# ========================
# Routes
# ========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term', '')
    names = []
    if term:
        names = df_anime[df_anime['Name'].str.contains(term, case=False, na=False)]['Name'].tolist()
    return jsonify(names)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    rec_type = data.get('recommendation_type')
    n = int(data.get('num_recommendations', 10))

    if rec_type == "item_based":
        anime_name = data.get('anime_name')
        recs = find_similar_animes(anime_name, n)
        return jsonify(recs.to_dict(orient='records'))

    return jsonify({"error": "Only item-based recommendation is supported right now."}), 400


# ========================
# Run the app
# ========================
if __name__ == '__main__':
    app.run(debug=True)
