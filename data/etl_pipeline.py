# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data and convert to dataframe
    INPUT: 
    messages_filepath: str, path to file
    categories_filepath: str, path to file

    OUTPUT:
    df : pandas dataframe"""
    # load messages dataset
    messages = pd.read_csv(messages_filepath, encoding='latin-1')

    # load categories dataset
    categories = pd.read_csv(categories_filepath, encoding='latin-1')

    # merge datasets
    df = messages.merge(categories, on='id', how='left')
    
    return df

def clean_data(df):
    """Clean data and transform category column into 36 individual columns
    INPUT: 
    df - pandas dataframe
    OUTPUT:
    df - cleaned dataframe """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    columns = []
    for col in categories.iloc[0]:
        col_fixed = col.split('-',1)[0]
        columns.append(col_fixed)

    categories.columns=columns

    #Convert category values to just numbers 0 or 1
    for column in categories:
       # set each value to be the last character of the string and convert it to integer
      categories[column] = categories[column].apply(lambda x: x.split('-',1)[1]).astype(int)

    #Replace categories column in df with new category columns

    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], sort=False, axis=1)

    #Drop duplicates
    df.drop_duplicates(inplace=True, keep='first')

    return df

def save_data(df, database_filepath):
    """Saves df to sql database
    INPUT:
    df - pandas dataframe
    database_filepath - sql data base path"""

    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df.to_sql('cat_messages', engine, index=False, if_exists = 'replace')

def main():
    "Load data, cleans it and saves it in SQL database"

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        df = load_data(messages_filepath, categories_filepath)

        df = clean_data(df)

        save_data(df, database_filepath)

        print('Data saved in {}'.format(database_filepath))

    else:
         print('Provide as arguments:'\
              'First argument: messages.csv filepath'\
              'Second argument: categories.csv filepath'\
              'Third argument: sql database filepath to save df'\
              '\n\nExample: python process_data.py '\
              'messages.csv categories.csv '\
              'nlp_messages.db')

if __name__ == '__main__':
    main()
