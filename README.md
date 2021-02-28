# Disaster Response Pipeline Project
A project to train a machine learning model to classify text into various topic categories to enable effective logging of messages related to disaster response needs. New messages can be dynamically classified through a web application.

## Project Organization
```
├── README.md                    <- The top-level README for developers using this project
├── app                          <- Web application files
│   ├── run.py                   <- Web app Python script
│   └── templates                <- Web app templates
│       ├── go.html
│       └── master.html
├── code                          <- Data processing and modeling files
│   ├── process_data.py           <- Data processing script
│   └── train_classifier.py       <- Model training script
├── data                          <- Data files
│   ├── processed                 <- Processed data
│   │   └── disaster_response.db  <- Processed SQLite database
│   └── raw
│       ├── categories.csv        <- Raw message categories data
│       └── messages.csv          <- Raw messages data
├── models                        <- Trained models directory
└── notebooks                     <- Practice notebooks
    ├── ETL\ Pipeline\ Preparation.ipynb  <- Data processing notebook
    └── ML\ Pipeline\ Preparation.ipynb   <- Model training notebook
```

## Data Files
- `data/raw/categories.csv` - Data file containing all of the possible categories of message data.
- `data/raw/messages.csv` - Data file containing the text message data that will primarily be used for modeling.

## Primary Code Files
- `code/process_data.py` - Takes in raw message and categorization data, merges them, cleans the data, then saves the processed data to an SQLite database in the `data/processed` directory
- `code/train_classifier.py` - Loads processed text data from SQLite database in `data/processed`, transforms it with count vectorization and TF-IDF, trains a Random Forest classification model and saves the trained model to the `models` directory.
- `app/run.py` - Runs the flask web app code that will display the web page where new messages can be entered and their predicted categories dynamically returned.

# Notebook Files
- `notebooks/ETL Pipeline Preparation.ipynb` - Notebook where the loading, cleaning and processing code was explored and tested.
- `notebooks/ML Pipeline Preparation.ipynb` - Notebook where the model training code was explored and tested.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python code/process_data.py data/raw/messages.csv data/raw/categories.csv disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python code/train_classifier.py data/processed/disaster_response.db models/trained_pipeline_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

