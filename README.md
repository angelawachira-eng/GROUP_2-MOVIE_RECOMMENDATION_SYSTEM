# **Hybrid Movie Recommendation System**

<img width="1274" height="735" alt="image" src="https://github.com/user-attachments/assets/316f137b-b852-415b-b71a-58c9c8009fae" />

Author: ***Group 2***

 Members: 
 
 Andrew Nyakiba
 
 Angela Wachira
          
 Bobbin Bodo
 
  Mercy Chepkoech
  
   Ted Mwenda
   
**Movie Recommendation System data**

***Tableau Link***: https://public.tableau.com/app/profile/ted.mutuma/viz/Phase4project_17772169395600/MovieRecommendationAnalysis?publish=yes

## 1. Project overview

This project aims to build a personalized hybrid movie recommendation system using collaborative filtering and content-based filtering on the MovieLens dataset. The system will learn from individual users' explicit ratings to predict which unwatched movies they are most likely to enjoy and return the top 5 personalized recommendations per user.



## 2. Business and Data Understanding

### 2.1 Stakeholder

The primary stakeholder is **Ziki**, a fictional Nairobi-based streaming
platform targeting East African users.

### 2.2 Business Problem

Streaming platforms offer huge catalogs, but without strong personalization, users face decision fatigue. Ziki offers a large catalog of movies, which can make it difficult for users to quickly find content that matches their preferences, leading to decision fatigue and reduced engagement. The goal of this project is to improve user experience by building a recommendation system that provides the top five personalized movie suggestions based on user ratings and content features.

### 2.3 Objectives
3.1 Primary Objective

- The main objective of this project is to build a hybrid recommendation system that can provide top 5 movie recommendations to a user based on prior preferences.

2.4 Secondary Objectives

- Identifying top genres by ratings.
  
- Building a collaborative filtering model for user-rating analysis.
  
- Building a content-based filtering model to handle the cold-start problem.
  
- Developing a user interface for recommendations.

### 2.5 Dataset Description

 | Dataset    | Description               | Purpose                              |
|------------|---------------------------|--------------------------------------|
| ratings.csv| User ratings for movies   | Collaborative filtering              |
| movies.csv | Movie titles and genres   | Translate movie IDs to titles and genres |
| tags.csv   | User-generated keywords   | Content-based filtering              |

## 3. Modeling

Four collaborative filtering algorithms were trained and compared using 5-fold cross-validation on the full ratings dataset. The rating scale was set to 0.5 – 5.0 using the Surprise library, with an 80/20 train-test split.

**Algorithms compared:**
- KNN Basic (item-based, cosine similarity)
- KNN With Means (item-based, cosine similarity)
- NMF (Non-negative Matrix Factorization)
- SVD (Singular Value Decomposition)

SVD achieved the lowest error on both RMSE and MAE and was selected for hyperparameter tuning using `GridSearchCV` over combinations of `n_factors` (50, 100), `n_epochs` (20, 30), `lr_all` (0.002, 0.005), and `reg_all` (0.02, 0.1).

**Content-based filtering** was implemented in parallel using TF-IDF vectorization over a combined feature string of each movie's genres and user-generated tags, with cosine similarity used to find similar movies.

**Collaborative filtering** recommends movies based on patterns in user behavior and rating similarities across the platform. It assumes that users with similar preferences are likely to enjoy similar movies. The final collaborative model selected was Tuned SVD, which learns latent relationships between users and movies to generate accurate personalized rating predictions.

**Hybrid Approach** system combines Content-Based Filtering and Collaborative Filtering to generate accurate, personalized movie suggestions.


A **cold-start fallback** was also built for brand-new users. It matches the user's keyword preferences against the movie tag index (Level 1), and falls back to Bayesian-scored genre-based recommendations if no tags match (Level 2).

## 4. Evaluation

### Model Performance

 | Model         | RMSE  | MAE   |
|---------------|-------|-------|
| KNN Basic     | ~0.97 | ~0.76 |
| KNN With Means| ~0.90 | ~0.69 |
| NMF           | ~0.91 | ~0.70 |
| **SVD (Tuned)**| **~0.87** | **~0.67** |

## 5. Conclusion

The hybrid recommendation system successfully meets Ziki's business objective of delivering relevant, personalized movie suggestions at scale. The tuned SVD model is the strongest individual algorithm tested, and its combination with content-based filtering makes the system robust across both existing and brand-new users.

**Key Findings:**

- **SVD Superiority**: Among all tested algorithms, the SVD (Singular Value Decomposition) model, fine-tuned via GridSearchCV, emerged as the most accurate predictor of user sentiment, achieving an RMSE of ~0.87.

- **Hybrid Advantage**: By blending SVD with Content-Based Similarity, we successfully mitigated the "filter bubble" effect. The hybrid model doesn't just predict what a user will rate highly; it also identifies why they might like it based on the movie's "DNA" (genres and tags).

- **Reliable Fallbacks**: Our implementation of Bayesian Weighted Scores for new users ensures that the platform remains trustworthy from day one, surfacing "Gold Standard" content instead of obscure, low-count titles.
  
 ## **Limitations to address in future iterations:**

Despite the success of our model, the following limitations should be noted:

1. Western Data Bias – The MovieLens dataset is heavily focused on Hollywood content and may not fully reflect East African audience preferences.

2. Explicit Data Constraints – The model uses only star ratings and excludes valuable behavioral signals like watch time, repeat views, and browsing activity.

3. Tag Sparsity – Many movies lack user-generated tags, limiting the effectiveness of content-based recommendations for some titles.

4. Static Time Analysis – Older and newer ratings are treated equally, even though user preferences change over time.
Computational Cost – GridSearchCV is resource-intensive and may not scale efficiently for very large user bases.



## **Business recommendations for Ziki:**

1. **Deploy Tuned SVD Model** – Use the tuned SVD model as the main recommendation engine for personalized homepage suggestions due to its strong accuracy and efficiency.
   
2. **Implement Keyword Onboarding** – Ask new users to select 3–5 preference keywords during sign-up to instantly generate relevant recommendations and reduce zero-watch sessions.
   
3. **Promote Hidden Gems** – Create recommendation categories for highly rated niche genres like Film-Noir and War to diversify viewing choices beyond mainstream titles.
   
4. **Retrain Weekly** – Update the SVD model weekly using fresh rating data to keep recommendations aligned with trends and seasonal preferences.
 
5. **Use Watch-Time Data** – Incorporate implicit signals such as watch duration and completion rates to better understand user interests beyond star ratings.



