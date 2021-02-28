import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import pickle

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """Load processed data from sqlite database

    Parameters:
    -----------
    database_filepath: Path to sqlite database

    Returns:
    --------
    X: Feature data
    y: Target labels
    category_names: List of all category labels"""

    # load data from database
    # conn = sqlite3.connect('../data/processed/disaster_response.db')
    conn = sqlite3.connect('data/processed/disaster_response.db')
    df = pd.read_sql_query('SELECT * FROM disaster_response', conn)
    conn.close()

    # Create feature and target data sets
    X = df['message']
    y = df.iloc[:, 4:]

    # Get list of category label names
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """Tokenize text with normalization, stripping and lemmatization.
    Parameters:
    -----------
    text: List of text documents
    
    Returns:
    --------
    clean_tokens: tokenized documents"""
    
    # Create tokens
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Clean and lemmatize tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """Creates the model pipeline with CountVectorizer, TF-IDF and RandomForestClassifier"""

    # Build pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('random_forest', RandomForestClassifier())])
    return pipeline
    
def evaluate_model(model, X_test, y_test, category_names):
    """Show metric results of model predictions

    Parameters:
    -----------
    model: Trained scikit-learn model
    X_test: Pandas df/series with test observations
    y_test: Pandas df/series set with test labels
    category_names: List of label names of target categories

    Returns:
    --------
    Prints Accuracy, F1 (Macro), Recall (Macro), Precision (Macro) scores and a 
    classification report for F1, Recall, and Precision for all categories
    """
    # Get model predictions
    y_pred = model.predict(X_test)

    print('Overall Accuracy:', accuracy_score(y_test, y_pred))
    print('Overall F1 (Macro):', f1_score(y_test, y_pred, average='macro'))
    print('Overall Recall (Macro):', recall_score(y_test, y_pred, average='macro'))
    print('Overall Precision (Macro):', precision_score(y_test, y_pred, average='macro'))
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Save trained model
    
    Parameters:
    -----------
    model: Trained scikit-learn model
    model_filepath: Output path for saved model

    Returns:
    --------
    Saves as pickle to designated location
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Runs the scripts that loads the processed data, builds the model, splits it into training
    and test DataFrames, evaluates it, then saves the trained model to a specified directory."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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