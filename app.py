import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# ---------------- UI ----------------
st.set_page_config(page_title="Netflix Recommender", layout="centered")

st.title("🎬 Netflix Recommendation System (NeuMF)")
st.caption("AI-powered movie recommendation using Neural Collaborative Filtering")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    user_map = np.load("user_mapping.npy", allow_pickle=True).item()
    movie_map = np.load("movie_mapping.npy", allow_pickle=True).item()
    ratings = pd.read_csv("ratings.csv")

    movies = pd.read_csv(
        "ml-1m/movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding='latin-1'
    )

    movie_titles = dict(zip(movies['movieId'], movies['title']))

    return user_map, movie_map, ratings, movie_titles

user2user_encoded, movie2movie_encoded, ratings, movie_id_to_title = load_data()

# ---------------- User Input ----------------
st.subheader("👤 Enter User ID")
user_id = st.number_input("User ID", min_value=1, max_value=6040, step=1)

top_k = st.slider("🎯 Number of recommendations", 1, 10, 5)

# ---------------- Recommendation Logic ----------------
if st.button("Recommend Movies"):
    try:
        user_encoded = user2user_encoded[user_id]

        # Movies already watched by user
        watched_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()

        # All movies
        all_movies = list(movie2movie_encoded.keys())

        # Filter unseen movies
        unseen_movies = list(set(all_movies) - set(watched_movies))

        # Encode unseen movies
        unseen_encoded = [movie2movie_encoded[x] for x in unseen_movies]

        user_array = np.array([user_encoded] * len(unseen_encoded))
        movie_array = np.array(unseen_encoded)

        # Predict
        predictions = model.predict([user_array, movie_array], verbose=0)

        # Get top K
        top_indices = predictions.flatten().argsort()[-top_k:]
        recommended_movie_ids = [unseen_movies[i] for i in top_indices]

        # ---------------- Display ----------------
        st.success("🎯 Top Recommendations for You")

        for movie in recommended_movie_ids[::-1]:
            st.markdown(f"""
            <div style="
                padding:10px;
                border-radius:10px;
                margin-bottom:10px;
                background-color:#262730;
                color:white;">
                🎬 {movie_id_to_title[movie]}
            </div>
            """, unsafe_allow_html=True)

    except KeyError:
        st.error("⚠️ Invalid User ID. Please enter a valid user.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
