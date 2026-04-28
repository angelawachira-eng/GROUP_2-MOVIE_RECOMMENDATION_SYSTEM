# Hybrid Movie Recommendation System

![Project Thumbnail](image/thumbnail.jpg)


## Project overview

This project builds a personalized hybrid movie recommendation system that combines collaborative filtering and content-based filtering on the MovieLens dataset. The system learns from individual users' explicit ratings to predict which unwatched movies they are most likely to enjoy, returning the top 5 personalized recommendations per user. It also addresses the cold-start problem for new users who have no rating history.


## Business and Data Understanding

### Stakeholder

The primary stakeholder is **Ziki**, a fictional Nairobi-based streaming
platform targeting East African users.

### Business Problem

Users struggle to find relevant content quickly due to large content
libraries, leading to reduced engagement.

### Objectives
- Identifying top genres by ratings\
- Building a collaborative filtering model for user-rating analysis\
- Building a content-based filtering model to handle the cold-start problem\
- Developing a user-interface for recommendations.

### Dataset Description

 | Dataset    | Description               | Purpose                              |
|------------|---------------------------|--------------------------------------|
| ratings.csv| User ratings for movies   | Collaborative filtering              |
| movies.csv | Movie titles and genres   | Translate movie IDs to titles and genres |
| tags.csv   | User-generated keywords   | Content-based filtering              |
## Modeling

Four collaborative filtering algorithms were trained and compared using 5-fold cross-validation on the full ratings dataset. The rating scale was set to 0.5 – 5.0 using the Surprise library, with an 80/20 train-test split.

**Algorithms compared:**
- KNN Basic (item-based, cosine similarity)
- KNN With Means (item-based, cosine similarity)
- NMF (Non-negative Matrix Factorization)
- SVD (Singular Value Decomposition)

SVD achieved the lowest error on both RMSE and MAE and was selected for hyperparameter tuning using `GridSearchCV` over combinations of `n_factors` (50, 100), `n_epochs` (20, 30), `lr_all` (0.002, 0.005), and `reg_all` (0.02, 0.1).
**Content-based filtering** was implemented in parallel using TF-IDF vectorization over a combined feature string of each movie's genres and user-generated tags, with cosine similarity used to find similar movies.

**The final Hybrid Score** blends both signals:

```
Hybrid Score = (SVD predicted rating × 0.70) + (Content similarity × 5 × 0.30)
```

A **cold-start fallback** was also built for brand-new users. It matches the user's keyword preferences against the movie tag index (Level 1), and falls back to Bayesian-scored genre-based recommendations if no tags match (Level 2).

## Evaluation

### Model Performance

 | Model         | RMSE  | MAE   |
|---------------|-------|-------|
| KNN Basic     | ~0.97 | ~0.75 |
| KNN With Means| ~0.90 | ~0.69 |
| NMF           | ~0.92 | ~0.71 |
| **SVD (Tuned)**| **~0.87** | **~0.67** |

## Conclusion

The hybrid recommendation system successfully meets Ziki's business objective of delivering relevant, personalized movie suggestions at scale. The tuned SVD model is the strongest individual algorithm tested, and its combination with content-based filtering makes the system robust across both existing and brand-new users.

**Key Findings:**

-   Hybrid systems improve recommendation quality\
-   SVD provides best personalization\
-   Tags help solve cold-start


**Business recommendations for Ziki:**

1. **Deploy the tuned SVD model** as the core recommendation engine — it outperforms all KNN variants and scales well to larger user bases.
2. **Onboard new users with keyword prompts** — the tag-based cold-start fallback immediately serves relevant picks before any ratings exist, reducing early drop-off.
3. **Retrain the model weekly** — user preferences shift over time; a stale model will underperform as the catalog grows.
4. **Monitor zero-watch sessions** — if a user browses but watches nothing, flag them for the cold-start fallback to re-engage them with fresh content.

**Limitations to address in future iterations:**

- The MovieLens dataset is Western-centric; taste patterns may differ for East African audiences, and locally relevant content data should be incorporated.
- Tags are sparse — only approximately 20% of movies have any user-generated keywords, which limits content-based signal quality.
- The model relies solely on star ratings; richer behavioural signals like watch time, repeat views, and search queries could significantly improve recommendation quality.



