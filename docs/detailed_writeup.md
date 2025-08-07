# All-NBA Selection Prediction: Detailed Project Write-Up

## Introduction  
Predicting All-NBA team selections has traditionally been a subjective exercise, influenced by a mix of statistics, reputation, and media narratives. This project uses machine learning techniques to create an objective, data-driven model that forecasts which NBA players are most likely to make the All-NBA teams in a given season. By analyzing multiple years of player performance data, the project identifies key metrics that correlate with selection and builds a reliable predictive framework.

---

## Project Motivation  
The All-NBA teams celebrate the league’s top performers, but the selection process is not always transparent or consistent. A rigorous predictive model can provide valuable insights for analysts, coaches, and fans by quantifying which stats matter most and reducing bias in the evaluation process. Additionally, the model offers a tool for furthering basketball analytics and decision-making workflows.

---

## Data Collection and Preparation  

### Data Sources  
The dataset aggregates player statistics from NBA seasons 2015 through 2024. It includes both traditional per-game stats (points, rebounds, assists) and advanced metrics (usage rate, effective field goal percentage, 3-point attempt rate, plus-minus ratings). The initial data was sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats).

### Data Cleaning  
Three raw datasets were used: end-of-season awards, per-game player statistics, and season-long advanced player stats. After loading, the data was filtered to include only seasons from 2015–2024 and players with at least 45 games played, which helped reduce the influence of statistical outliers. Per-game and advanced stats were aggregated at the player level to produce one observation per eligible player. After cleaning, the merged dataset contained 3,198 rows and 22 columns. These steps are fully documented in the `01_data_prep.ipynb` notebook.

### Feature Engineering  
The target variable, `all_nba`, was engineered by checking whether a player was assigned to any of the three All-NBA teams (1st, 2nd, or 3rd). If so, their `all_nba` value was set to 1; otherwise, it was 0.

---

## Modeling Approach  

### Data Preprocessing  
Features and the target variable were defined and split into training and test sets, with the test set comprising 25% of the data. Feature normalization was applied to ensure all numerical inputs were on the same scale for models that required it.

### Baseline Models  
Multiple classification models were tested to establish baseline performance: Logistic Regression, Random Forest, k-Nearest Neighbors, Support Vector Machines, and XGBoost. Each model underwent randomized hyperparameter search with cross-validation.

### Addressing Class Imbalance  
Because All-NBA players represent a small portion of the dataset, class imbalance was a central challenge. Stratified splitting ensured balanced class representation in training and test sets. During modeling, both **class weighting** and **SMOTE (Synthetic Minority Oversampling Technique)** were tested depending on model compatibility.

### Threshold Optimization  
Rather than using the default classification threshold of 0.5, thresholds were tuned to maximize F1-score, balancing precision and recall. This yielded better performance for this imbalanced classification task.

### Model Selection  
Two modeling pipelines were compared, with one using SMOTE and one using class weighting. Logistic Regression emerged as the best-performing model in both.  
- SMOTE-based logistic regression (C = 0.619) achieved an average precision of **0.899**  
- Class-weighted logistic regression (C = 1.03) achieved an average precision of **0.900**.

Further threshold tuning was done using cross-validation to maximize F1-score:  
- SMOTE pipeline: **CV mean F1 = 0.833**  
- Class-weighted pipeline: **CV mean F1 = 0.830**

Due to comparable performance and lower computational cost, the **class-weighted logistic regression** model was chosen as the final model. Both the model and the test sets were serialized for final evaluation.

---

## Evaluation and Results  

### Metrics  
The model was evaluated using:  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Average Precision  
- ROC-AUC  

Confusion matrices illustrated performance, and a precision-recall curve with a threshold marker showed trade-offs between precision and recall across thresholds.

### Final Model Performance  
The final model, which was a class-weighted logistic regression with a tuned threshold, performed well:  
- **F1 Score:** 0.786  
- **ROC-AUC:** 0.926  

The model effectively identified a large portion of actual All-NBA selections while maintaining a relatively low false positive rate. It provides an objective tool for use by fans, analysts, and media alike.

---

## Deployment: Streamlit Web Application  

### App Overview  
A Streamlit web application was built for public use. It allows users to input custom player stats and receive real-time predictions of All-NBA selection probability.

### User Experience  
The interface is simple and intuitive, designed to be accessible to basketball fans, analysts, and data scientists. Users can explore how changes in player statistics affect predicted outcomes.

### Technical Details  
The app uses the trained logistic regression model and optimized threshold. It is deployed using Streamlit Cloud and can be accessed [here](https://all-nba-predictor.streamlit.app).

---

## Challenges and Lessons Learned  

- **Data Quality:** Preprocessing was crucial, removing edge cases like low-minute players significantly improved modeling stability.  
- **Class Imbalance:** The imbalance between All-NBA players and non-selections required experimentation with balancing strategies and evaluation metrics.  
- **Feature Selection:** Identifying and eliminating redundant variables helped ensure cleaner model training.  
- **Deployment Experience:** Creating a Streamlit app offered hands-on experience with real-world deployment and user-facing interfaces.

---

## Future Directions  

- Integrate newer player tracking and impact-based metrics (e.g., RAPTOR, LEBRON).  
- Experiment with neural networks or ensemble stacking methods.  
- Automate data ingestion pipelines for up-to-date predictions.  
- Add model explainability features such as SHAP or LIME for transparency.

---

## How to Use This Project  

1. Explore the notebooks in the `/notebooks` folder to follow the complete pipeline from raw data to model evaluation.  
2. Run the Streamlit app locally (`app.py`) or visit the deployed version to interact with the model.  
3. Use the codebase as a starting point for further experimentation or integration into larger analytics platforms.

---

## Author  
**Jack Ewings**


