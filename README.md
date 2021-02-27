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

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python code/process_data.py data/raw/messages.csv data/raw/categories.csv disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python code/train_classifier.py data/processed/disaster_response.db models/trained_pipeline_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

