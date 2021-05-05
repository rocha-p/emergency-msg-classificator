# Disaster Response Pipeline Project
### By Pedro Rocha
### 05/05/2021

## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. 
This project aims to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.
This model is useful, for example, to forward messages to the corresponding organization according to their classification in real time, and thus provide help as quickly and efficiently as possible.

## Content
- Data
  - etl_pipeline.py: reads in the data, cleans and stores it in a SQL database. Basic usage is python etl_pipeline.py MESSAGES_DATA CATEGORIES_DATA NAME_FOR_DATABASE
  - categories.csv and disaster_messages.csv (dataset)
  - nlp_pipeline.db: created database from transformed and cleaned data.
- Models
  - train.py: includes the code necessary to load data, transform it using natural language processing, run a machine learning model using GridSearchCV and train it. Basic usage is python train.py DATABASE_DIRECTORY SAVENAME_FOR_MODEL  
- App
  - run.py: Flask app and the user interface used to predict results and display them.
  - templates: folder containing the html templates

## Example:
> python etl_pipeline.py messages.csv categories.csv nlp_messages.db

> python train.py ../data/nlp_messages.db model.pkl

> python run.py

## Screenshots
This is the frontpage:
![Alt text](https://github.com/rocha-p/emergency-msg-classificator/blob/main/disasters-front-page.png)

By inputting a word, you can check its category:
![Alt text](https://github.com/rocha-p/emergency-msg-classificator/blob/main/disasters-results.png)

## About
This project was prepared as part of the Udacity Data Scientist nanodegree. The data was provided by Figure Eight. 
