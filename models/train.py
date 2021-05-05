# import libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """load data from sql database and return X, y
    INPUT: database_filepath - path to sql database
    OUTPUT: X, y """

    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql('SELECT * FROM cat_messages', engine)
    X = df.message
    y = df.iloc[:,4:]
    return X, y


def tokenize(text):

    """Normalize, tokenize, and lemmatize text. Return words ready to vectorize
    INPUT:
    text - str

    OUTPUT:
    list of words ready to vectorize - List"""

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    words = [lemmatizer.lemmatize(words).strip() for words in tokens if words not in stopwords.words("english")]
    
    return tokens

def build_model():
    """vectorize and transform data, gridsearch for best parameters and returns a ML classifier model
    
    OUTPUT: ML model - MultiOutputClassifier(RandomForest)"""

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
         'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters, cv=2, verbose=3, n_jobs = -1)
    
    return cv

def display_results(y_test, y_pred):
    """shows f1_score, recall and precision for each category
    INPUT:
    y_test - test data 
    y_pred - predicted data

    OUTPUT:
    scores dataframe"""


    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    for i, cat in enumerate(y_test.columns):
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,i], average='weighted')
        results.at[i+1, 'Category'] = cat
        results.at[i+1, 'f_score'] = f_score
        results.at[i+1, 'precision'] = precision
        results.at[i+1, 'recall'] = recall
        
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    return results

def save_model(model, model_filepath):
    """Export model as a picke file
    INPUT:
    model - ML model
    model_filepath - path to save picke file"""

    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """Load the data, split in test-train, run the model, show the socore and save the model in a picle file"""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        #load data
        print("Loading data...")
        X, y = load_data()
    
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X,y)
    
        #initialize model
        print("Initializing model...")
        model = build_model()
    
        #train model
        print('Training model...')
        model.fit(X_train, y_train)
    
        #predict y values
        y_pred = model.predict(X_test)
    
        #display f1 score, precision and recall for each output category
        print('Evaluating model...')
        display_results(y_test, y_pred)

        #save model as pickle file
        print('Saving model...')
        save_model(model, model_filepath)

        print("Model saved!")

    else:
        print('Provide arguments: FIRST: filepath of the SQL database '\
              'SECOND: filepath of the pickle file to save the model'\
              '\n\nExample: python train.py ../data/nlp_messages.db model.pkl')

if __name__ == '__main__':
    main()
