import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load data
# -------------------------
ratings = pd.read_csv("Data/ml-latest-small/ratings.csv")
movies = pd.read_csv("Data/ml-latest-small/movies.csv")

# Movie stats (for fallback)
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

movies = movies.merge(movie_stats, on='movieId', how='left')

# -------------------------
# Build user-item matrix
# -------------------------
user_item_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# Compute similarity
user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# -------------------------
# Helper functions
# -------------------------
def get_movie_id(title):
    return movies[movies['title'] == title]['movieId'].values[0]

def recommend_for_new_user(user_ratings, n=5):
    # Create user vector
    user_vector = pd.Series(0, index=user_item_matrix.columns)

    for title, rating in user_ratings.items():
        movie_id = get_movie_id(title)
        if movie_id in user_vector.index:
            user_vector[movie_id] = rating

    # Compute similarity
    similarities = cosine_similarity(
        [user_vector],
        user_item_matrix
    )[0]

    similar_users = pd.Series(
        similarities,
        index=user_item_matrix.index
    ).sort_values(ascending=False)

    top_users = similar_users.iloc[1:11]

    # Weighted ratings
    weighted_ratings = user_item_matrix.loc[top_users.index].T.dot(top_users)

    # Remove already rated movies
    for title in user_ratings:
        movie_id = get_movie_id(title)
        if movie_id in weighted_ratings.index:
            weighted_ratings.drop(movie_id, inplace=True)

    top_movies = weighted_ratings.sort_values(ascending=False).head(n).index

    return movies[movies['movieId'].isin(top_movies)][['title', 'avg_rating']]

def hybrid_recommend(user_ratings, n=5):
    if len(user_ratings) >= 5:
        return recommend_for_new_user(user_ratings, n)
    else:
        return movies.sort_values(
            ['avg_rating', 'num_ratings'],
            ascending=False
        ).head(n)[['title', 'avg_rating']]

# -------------------------
# Poster function
# -------------------------
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

# Session state
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []

# -------------------------
# Search
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
            if st.button("Add", key=f"add_{movie}"):
                if movie not in st.session_state.selected_movies:
                    st.session_state.selected_movies.append(movie)

# -------------------------
# Selected movies
# -------------------------
st.write("### ✅ Selected Movies")

if st.session_state.selected_movies:
    for movie in st.session_state.selected_movies:
        col1, col2, col3 = st.columns([1, 4, 1])

        poster_url = fetch_poster(movie)

        with col1:
            if poster_url:
                st.image(poster_url, width=60)

        with col2:
            st.write(movie)

        with col3:
            if st.button("❌", key=f"remove_{movie}"):
                st.session_state.selected_movies.remove(movie)
                st.rerun()
else:
    st.info("No movies selected yet")

# -------------------------
# Ratings
# -------------------------
st.write("### ⭐ Rate Selected Movies")

user_ratings = {}

for movie in st.session_state.selected_movies:
    rating = st.slider(
        f"Rate {movie}",
        0.5, 5.0, 3.0,
        key=f"rate_{movie}"
    )
    user_ratings[movie] = rating

# -------------------------
# Recommendations
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