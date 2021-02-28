import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categores date files and merge them in to a single DataFrame

    Parameters:
    -----------
    messages_filepath: path to messages.csv file
    categories_filepath: path to categories.csv file

    Returns:
    --------
    df: Pandas DataFrame with messages and categories merged 
    """
    # Load messages and categories files, dropping duplicate rows in each
    messages = pd.read_csv(messages_filepath).drop_duplicates()
    categories = pd.read_csv(categories_filepath).drop_duplicates()

    # Merge categories and duplicates
    df = messages.merge(categories, how='inner')

    return df

def clean_data(df):
    """
    Clean DataFrame, splitting categories into columns and merging back with messages in df

    Parameters:
    -----------
    df: Pandas DataFrame

    Returns:
    --------
    df: Pandas DataFrame 
    """

    # Create categories df, splitting categories strings into individual columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace categories column in df with new category columns
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)

    # Change related column to binary, as there are instances of 2 instead of just 1 and 0
    df.replace({'related':{2:1}}, inplace=True)

    return df

def save_data(df, database_filename):
    """Save processed data to an SQLite database
    
    Parameters:
    -----------
    df: DataFrame to be stored in SQLite database
    database_filename: Path and name of SQLite database to be created

    Output:
    --------
    SQLite database saved to specified directory 
    """

    engine = create_engine('sqlite:///data/processed/' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')  


def main():
    """Main script to be run when process_data.py is executed. 
    Runs the data loading, data cleaning, and data save functions"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python code/process_data.py '\
              'data/raw/messages.csv data/raw/categories.csv '\
              'disaster_response.db')


if __name__ == '__main__':
    main()