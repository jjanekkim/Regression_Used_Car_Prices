# Used Car Regression Project

## Table of Contents
* [Introduction](#introduction)
* [Take Away](#take-away)
* [Dependencies](#dependencies)
* [Project Structure](#project-structure)

## Introduction
This is a Kaggle Playground Competition.

Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.
Goal: The goal of this competition is to predict the price of used cars based on various attributes.

## Take Away

I participated in this regression competition to keep my skills fresh and stay current with Python coding. During the project, I noticed that feature engineering was not improving the predictions; instead, it was worsening the scores.

After some research, I discovered that the issue stemmed from the data being synthetically generated by a computer. Since the computer-generated data followed specific patterns, the algorithms detected these patterns rather than genuine trends. Consequently, feature engineering confused the algorithms, leading to poorer predictions.

Despite this challenge, I enjoyed the competition, particularly the numerous steps involved in data validation. Data cleaning was extensive, addressing issues from coherent string cases to model and brand mismatches. I used dictionaries to ensure that brands and models were correctly matched.

## Dependencies
This project is created with:
- Python version: 3.11.3
- Pandas version: 2.0.1
- Numpy version: 1.24.3
- Matplotlib version: 3.7.1
- Seaborn version: 0.12.2
- Re version: 2.2.1
- Scikit-learn package version: 1.2.2
- Catboost version: 1.2.7

## Project Structure
- Data: Access the dataset [here](https://www.kaggle.com/competitions/playground-series-s4e9/overview).
- **dashboard.py**: Python script for streamlit dashboard.
- **feature_engineering**: Jupyter notebook for feature engineering.
- **pipeline**: Python consisting the functions to clean, validate, preprocessing, and training data.
- **pipeline_testing**: Jupyter notebook for testing pipeline (function packages).
- **test_data**: Jupyter notebook working on the test data.
- **validation**: Jupyter notebook of cleaning and validating the data.
- **visualization**: Jupyter notebook with plotly visualizations.
