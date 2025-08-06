# All-NBA-Prediction

## Overview

This project aims to build a machine learning model that predicts which NBA players will be selected for the All-NBA teams each season. Using player performance statistics over recent years, the model identifies the key factors that influence selection and creates a predictive framework that can assist analysts, teams, and fans in understanding player impact.

## Motivation

The All-NBA teams highlight the best players in the league each year, but the selection process can be subjective. By applying data-driven techniques, this project seeks to provide an objective way to forecast selections based on measurable performance metrics.

## Data

The dataset includes player stats spanning multiple NBA seasons (2015-2024), containing traditional and advanced metrics such as points per game, shooting percentages, usage rate, rebounds, assists, and advanced plus-minus ratings. The dataset is cleaned and processed in the `01_data_prep.ipynb` notebook.

## Approach

- **Data Preparation:** Cleaning and feature engineering to create relevant inputs for modeling.
- **Baseline Models:** Tried multiple classifiers (Logistic Regression, Random Forest, KNN, SVM, XGBoost) with hyperparameter tuning via randomized search.
- **Handling Class Imbalance:** Tested class weighting and SMOTE oversampling to address the imbalance between All-NBA selections and non-selections.
- **Threshold Optimization:** Instead of default 0.5 threshold, the decision threshold was tuned to maximize F1-score for better balance between precision and recall.
- **Final Model:** Class-weighted Logistic Regression with a tuned threshold was selected due to its performance and efficiency.

## Results

The final model achieved strong recall and precision, correctly identifying a high proportion of All-NBA selections while maintaining a low false positive rate. Detailed evaluation metrics and confusion matrices are presented in the `03_final_model_evaluation.ipynb` notebook.

## Project Structure

- `data/` - Raw and processed datasets  
- `models/` - Saved trained models and threshold values  
- `notebooks/` - Jupyter notebooks for data prep, modeling, and evaluation  
- `src/` - Supporting scripts   
- `requirements.txt` - Python dependencies  

## How to Use

Run the notebooks in order for a step-by-step walkthrough of the entire modeling process, from raw data to final model evaluation.

---

*Created by Jack Ewings*
