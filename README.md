# Disaster Response Pipeline
Udacity Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions to implement the project](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
You will have to install the following libraries for the project:
* numpy
* pandas 
* sqlalchemy 
* re
* nltk
* sklearn
* pickle
* json
* plotly
* flask

And the following packages need to be downloaded for nltk:
punkt, wordnet, stopwords and averaged_perceptron_tagger

## Project Motivation<a name="motivation"></a>
Following a disaster, there are a number of different problems that may arise. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, so predictive modeling can help classify different messages more efficiently and so prompt action can be taken.

In this project my aim was to analyze disaster data from Figure Eight organisation to build a model for an API that classifies disaster messages. It includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## File Descriptions<a name="files"></a>
There are three folders in this repository:

1) data
* disaster_categories.csv: csv file containing data of the labelled categories for id
* disaster_messages.csv: csv file containing text message data for id 
* process_data.py: ETL pipeline script to read the dataset, clean the data, and then store it in a SQLite database
* DisasterResponse.db: SQLite database containing cleaned and merged data from messages and categories files
2) models
* train_classifier.py: ML pipeline script that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification)
* classifier.pkl: Final trained model
3) app
* run.py: Python script run the web application using Flask
* templates: contains template html files
  - master.html: main page of web app
  - go.html: classification result page of web app

## Instructions to implement the project<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

