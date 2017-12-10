# Question-Answering-NNs

## Using a new dataset
This project is aimed to work on datasets with the following columns(separated by tabs):
- question: String of any length without \n, \t
- answer: same as above
- context: same as above
- value: 1 or 0 expressing whether given the question the answer is correct

Given a dataset the recommendation is to place it in the following manner:
-create a "dataset_name" folder in the root of the project
-create 2 folders in it called "data" and "structures"
-place the dataset in a file called "datum.txt" in the "data" folder

./
  dataset_name
    structures
    data
      datum.txt

## How to train/use the network
- In order to split the dataset into training and testing run splitDatum.py
- For training a model run trainModel.py (remember to call the preprocessData function at least once on a new dataset. The process might take a while(20-30 mins) as it converts every word to it's lemma.
- For checking the results of your model run checkModel.py (remember to use the name of the weights file that you want to test)


## Requirements
python 3.5
tensorflow
keras
pandas
numpy
nltk
