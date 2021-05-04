# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv("messages.csv", encoding='latin-1')

# load categories dataset
categories = pd.read_csv("categories.csv", encoding='latin-1')

# merge datasets
df = messages.merge(categories, on='id', how='left')

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

#Save the clean dataset into an sqlite database
engine = create_engine('sqlite:///nlp_messages.db')
df.to_sql('cat_messages', engine, index=False, if_exists = 'replace')