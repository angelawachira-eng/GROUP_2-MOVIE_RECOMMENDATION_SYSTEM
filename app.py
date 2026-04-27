import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD

# -------------------------
# Load data
# -------------------------
ratings = pd.read_csv("Data/ml-latest-small/ratings.csv")
movies = pd.read_csv("Data/ml-latest-small/movies.csv")

# Create avg_rating for fallback
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

movies = movies.merge(movie_stats, on='movieId', how='left')

# -------------------------
# Train model (SVD)
# -------------------------
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

# -------------------------
# Helper functions
# -------------------------
def get_movie_id(title):
    return movies[movies['title'] == title]['movieId'].values[0]

def create_user_profile(user_ratings):
    user_data = []
    for title, rating in user_ratings.items():
        movie_id = get_movie_id(title)
        user_data.append((999999, movie_id, rating))
    return user_data

def recommend_for_new_user(user_ratings, n=5):
    user_data = create_user_profile(user_ratings)
    rated_movie_ids = [m for (_, m, _) in user_data]

    all_movies = movies['movieId'].unique()
    unseen_movies = [m for m in all_movies if m not in rated_movie_ids]

    predictions = []

    for movie_id in unseen_movies:
        pred = model.predict(999999, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = [m[0] for m in predictions[:n]]

    return movies[movies['movieId'].isin(top_movies)][['title', 'avg_rating']]

def hybrid_recommend(user_ratings, n=5):
    if len(user_ratings) >= 5:
        return recommend_for_new_user(user_ratings, n)
    else:
        return movies.sort_values(['avg_rating', 'num_ratings'], ascending=False).head(n)[['title', 'avg_rating']]


import requests

API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data
def fetch_poster(title):
    clean_title = title.split('(')[0].strip()
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={clean_title}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass

    return None
# -------------------------
# Streamlit UI
# -------------------------

st.title("🎬 Ziki Movie Recommender")
st.write("Search and add movies, then rate at least 5 to get recommendations")

# -------------------------
# SESSION STATE (store selections)
# -------------------------
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []

# -------------------------
# SEARCH BAR
# -------------------------
search_query = st.text_input("🔍 Search for a movie")

movie_titles = movies['title'].unique()

if search_query:
    filtered_movies = [
        m for m in movie_titles
        if search_query.lower() in m.lower()
    ][:10]

    st.write("### 🎬 Search Results")

    for movie in filtered_movies:
        poster_url = fetch_poster(movie)

        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if poster_url:
                st.image(poster_url, width=80)

        with col2:
            st.write(movie)

        with col3:
            if st.button("Add", key=movie):
                if movie not in st.session_state.selected_movies:
                    st.session_state.selected_movies.append(movie)

# -------------------------
# SHOW SELECTED MOVIES
# -------------------------
st.write("### ✅ Selected Movies")

if st.session_state.selected_movies:
    st.write(st.session_state.selected_movies)
else:
    st.info("No movies selected yet")

# -------------------------
# RATINGS SECTION
# -------------------------
st.write("### ⭐ Rate Selected Movies")

user_ratings = {}

for movie in st.session_state.selected_movies:
    rating = st.slider(f"Rate {movie}", 0.5, 5.0, 3.0, key=f"rate_{movie}")
    user_ratings[movie] = rating

# -------------------------
# RECOMMEND BUTTON
# -------------------------
if st.button("Get Recommendations"):

    if len(user_ratings) < 5:
        st.warning("Please rate at least 5 movies")

    else:
        recs = hybrid_recommend(user_ratings)

        st.write("### 🎯 Your Recommendations")

        cols = st.columns(5)

        for i, (_, row) in enumerate(recs.iterrows()):
            col = cols[i % 5]

            title = row['title']
            poster_url = fetch_poster(title)

            with col:
                if poster_url:
                    st.image(poster_url)
                st.caption(title)