import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    '''
    Working:
        Load cleaned data from SQLite database as dataframe
        and return Message Text, Output Categories dataframe and Category names (Labels)
    Input Parameters:
        database_filepath: File path to SQLite database
    Returns:
        X: Message Text dataframe
        Y: Output Categories dataframe
        category_names: Category names (Labels)
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    Working:
        Takes the text to normalize case, remove punctuation,
        tokenize, lemmatize and remove english stop words
    Input:
        text: Message text
    Output:
        tokens: Message text after performing the above operations
    '''
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

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

def build_model():
    '''
    Working:
        Build a ML pipeline using: 
        Transformers:
            Feature extraction from text using CountVectorizer(tokenizer=tokenize), TfidfTransformer() 
            Creating custom feature of text length
        Predictor:
            Making MultiOutputClassifier() predictior with AdaBoostClassifier() 
        Then using grid search to find best parameters
    Returns:
        GridSearchCV object (model)
    '''
    # Make a machine learning pipeline:
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('length', TextLength()),
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Dictionary of parameters for grid search 
    parameters = {'features__text_pipeline__vect__ngram_range':[(1,2),(2,2)],
                 'clf__estimator__n_estimators':[50, 100]
             }
    
    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Working:
        Evaluate model performance using test data
        Print the f1 score, precision and recall for each category
        Also print the Average F1-Score of all output categories
    Input Parameters: 
        model: Model to be evaluated
        X_test: Test dataframe (features)
        Y_test: True labels dataframe for Test data
        category_names: Label names 
    '''
    
    predictions = model.predict(X_test)
    predictions = pd.DataFrame(predictions, columns = Y_test.columns)
    
    f1_scores=[]
    # Print the f1 score, precision and recall for each category
    for column in category_names:
        print("Output category:", column)
        print(classification_report(Y_test[column], predictions[column]))
        f1_scores.append(f1_score(Y_test[column], predictions[column], average='macro'))
        
    # Print the Average F1-Score of all output categories    
    print("Average F1-Score of all output categories:", sum(f1_scores)/len(f1_scores))

def save_model(model, model_filepath):
    '''
    Working:
        Save a model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: Path where the model will be saved
    '''
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()