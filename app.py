import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(layout="wide")

# -------------------------
# CUSTOM CSS (Netflix style)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #141414;
    color: white;
}
.movie-card {
    transition: transform 0.3s ease;
}
.movie-card:hover {
    transform: scale(1.08);
}
.badge {
    background-color: #e50914;
    padding: 2px 6px;
    margin: 2px;
    border-radius: 5px;
    font-size: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
ratings = pd.read_csv("Data/ml-latest-small/ratings.csv")
movies = pd.read_csv("Data/ml-latest-small/movies.csv")

# -------------------------
# SAFE MOVIE ID FUNCTION
# -------------------------
def get_movie_id(title):
    result = movies[movies['title'] == title]
    if result.empty:
        return None
    return result['movieId'].values[0]

# -------------------------
# CREATE USER-MOVIE MATRIX
# -------------------------
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

cosine_sim = cosine_similarity(user_movie_matrix.T)

movie_ids = user_movie_matrix.columns
cosine_df = pd.DataFrame(cosine_sim, index=movie_ids, columns=movie_ids)

# -------------------------
# SAFE API KEY HANDLING
# -------------------------
try:
    API_KEY = st.secrets["TMDB_API_KEY"]
except:
    API_KEY = None

@st.cache_data
def fetch_poster(title):
    if not API_KEY:
        return None

    clean_title = title.split('(')[0].strip()
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={clean_title}"

    try:
        data = requests.get(url, timeout=5).json()
        if data.get('results'):
            poster = data['results'][0].get('poster_path')
            if poster:
                return f"https://image.tmdb.org/t/p/w500{poster}"
    except:
        return None

    return None

# -------------------------
# RECOMMENDATION FUNCTIONS
# -------------------------
def recommend_similar(movie_title, n=10):
    movie_id = get_movie_id(movie_title)
    if movie_id is None or movie_id not in cosine_df.columns:
        return pd.DataFrame()

    similar_scores = cosine_df[movie_id].sort_values(ascending=False)[1:n+1]
    return movies[movies['movieId'].isin(similar_scores.index)]

def recommend_popular(n=10):
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    merged = movies.merge(movie_stats, on='movieId')

    return merged.sort_values(
        ['avg_rating', 'num_ratings'],
        ascending=False
    ).head(n)

# ✅ NEW: REAL RATING-BASED RECOMMENDER
def recommend_from_ratings(user_ratings, n=10):
    user_vector = pd.Series(0.0, index=movie_ids)

    for movie, rating in user_ratings.items():
        movie_id = get_movie_id(movie)
        if movie_id and movie_id in user_vector.index:
            user_vector[movie_id] = float(rating)

    similarity = cosine_similarity([user_vector], user_movie_matrix)[0]
    similar_users = np.argsort(similarity)[::-1][1:10]

    recs = user_movie_matrix.iloc[similar_users].mean().sort_values(ascending=False)

    # remove already rated movies
    rated_ids = [get_movie_id(m) for m in user_ratings.keys()]
    recs = recs.drop(rated_ids, errors='ignore')

    return movies[movies['movieId'].isin(recs.head(n).index)]

# -------------------------
# UI TITLE
# -------------------------
st.title("Ziki Movies Recommendation System")

# -------------------------
# MODE SWITCH (SIDEBAR)
# -------------------------
mode = st.sidebar.radio(
    "Choose Mode",
    ["🔍 Search Movie", "⭐ Rate Movies"]
)

# -------------------------
# SESSION STATE
# -------------------------
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# =========================================================
# 🔍 SEARCH MODE
# =========================================================
if mode == "🔍 Search Movie":

    search = st.sidebar.text_input("Search for a movie")

    if search:
        results = movies[movies['title'].str.contains(search, case=False, na=False)].head(10)

        st.write("### Results")

        cols = st.columns(5)

        for i, (_, row) in enumerate(results.iterrows()):
            col = cols[i % 5]
            title = row['title']
            genres = row['genres'].split("|")

            with col:
                poster = fetch_poster(title)

                st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                if poster:
                    st.image(poster)

                st.caption(title)

                for g in genres[:2]:
                    st.markdown(f"<span class='badge'>{g}</span>", unsafe_allow_html=True)

                # ✅ FIXED: unique key + no duplicates
                if st.button("Select", key=f"search_{i}"):
                    if title not in st.session_state.selected_movies:
                        st.session_state.selected_movies.append(title)

                st.markdown('</div>', unsafe_allow_html=True)

    # SHOW RECOMMENDATIONS
    if st.session_state.selected_movies:
        selected = st.session_state.selected_movies[-1]

        st.write(f"### 🎯 Because you liked: {selected}")

        with st.spinner("Finding similar movies..."):
            recs = recommend_similar(selected)

        cols = st.columns(5)

        for i, (_, row) in enumerate(recs.iterrows()):
            col = cols[i % 5]
            title = row['title']
            genres = row['genres'].split("|")

            with col:
                poster = fetch_poster(title)

                if poster:
                    st.image(poster)

                st.caption(title)

                for g in genres[:2]:
                    st.markdown(f"<span class='badge'>{g}</span>", unsafe_allow_html=True)

# =========================================================
# ⭐ RATING MODE
# =========================================================
else:

    st.sidebar.write("### 🎯 Rate Movies")

    movie_titles = movies['title'].unique()

    selected = st.sidebar.multiselect("Choose movies", movie_titles)

    for movie in selected:
        rating = st.sidebar.slider(movie, 0.5, 5.0, 3.0, key=f"rate_{movie}")
        st.session_state.user_ratings[movie] = rating

    # SHOW SELECTED MOVIES
    st.write("### ⭐ Your Selected Movies")

    if st.session_state.user_ratings:
        for movie, rating in list(st.session_state.user_ratings.items()):
            col1, col2 = st.columns([4,1])

            with col1:
                st.write(f"🎬 {movie} — ⭐ {rating}")

            with col2:
                if st.button("❌", key=f"remove_{movie}"):
                    del st.session_state.user_ratings[movie]
                    st.rerun()
    else:
        st.write("No movies selected yet.")

    # GET RECOMMENDATIONS
    if st.sidebar.button("Get Recommendations"):

        if len(st.session_state.user_ratings) < 3:
            st.warning("Rate at least 3 movies")
        else:
            with st.spinner("Finding best movies for you..."):
                recs = recommend_from_ratings(st.session_state.user_ratings)

            st.write("### 🎯 Recommended For You")

            cols = st.columns(5)

            for i, (_, row) in enumerate(recs.iterrows()):
                col = cols[i % 5]
                title = row['title']
                genres = row['genres'].split("|")

                with col:
                    poster = fetch_poster(title)

                    if poster:
                        st.image(poster)

                    st.caption(title)

                    for g in genres[:2]:
                        st.markdown(f"<span class='badge'>{g}</span>", unsafe_allow_html=True)