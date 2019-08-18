import json
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


class TextLength(BaseEstimator, TransformerMixin):
    '''
    Working:
        Custom made transformer class for creating text length feature to be used in ML Pipeline
    Returns:
        Length of the text documents
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        '''
    Working:
        Function to find text length 
    Input Parameter:
        x: text column to be transformed 
    Returns:
        Length of the text documents in the text column
    '''
        return np.array([len(text) for text in x]).reshape(-1, 1)

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Count of the categories
    categories_count = df.iloc[:,4:].sum().sort_values(ascending=False)
    # Percentage of all messages labelled with the category
    categories_mean = df.iloc[:,4:].mean().sort_values(ascending=False)*100
    
    categories_name = list(categories_count.index)
    
    # Count number of categories per message
    df_cat = df.copy()
    df_cat['categories_num'] = df_cat.iloc[:, 3:].sum(1)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_name,
                    y=categories_count
                )
            ],

            'layout': {
                'title': 'Count of the categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_name,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Percentage of all messages labelled with the category',
                'yaxis': {
                    'title': "Percentage(%)"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x = df_cat['categories_num']
                    
                )
            ],

            'layout': {
                'title': 'Distribution of number of categories per message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of categories per message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()