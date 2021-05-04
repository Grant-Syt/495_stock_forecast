# COMP-495-S21
Time series machine learning with TensorFlow and Python for my mentored research

Intro:
This project uses machine learning on stock data and focuses on the the analysis
and manipulation of the data, the detailed analysis of the machine learning test
results, and the results of applying multiple approaches to machine learning.

Running the code:
The currently saved state of the notebook can be viewed on Github by clicking on the
stock_forecast file. Jupyter Notebooks can be run in Jupyter Lab or online via Google
Colab. If you run this on your local computer, note that you will need to have all
required packages installed in your environment. You can start by downloading Anaconda
and installing the packages in a new environment.

Avoiding errors:
The notebook can be run from top to bottom. Note that some code takes a significant
amount of time to run. When running things selectively or out of order some variables
may not be defined or may have unexpected values. You can fix these issues by running
the code block that defines the variable or sets it to a desired value.

Using additional stocks:
I included some stocks in the repository that can be accesed on demand. If you want to
add new stocks you can use the stock_converter notebook file. This notebook takes files
from the dataset I used and formats them for use with stock_forecast. For it to work
properly you will need to have the dataset I used in the proper directory. I used this
dataset: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs.
I put the data in COMP-495-S21/data/stock_data/, which contained the files "Data", "ETFs",
and "Stocks" from the dataset.

Future improvements:
I spent most of my time writing code that analyses the stocks and the success of the
machine learning model. Now that those are taken care of, the underlying machine learning
method could be changed, and those results would be easily seen in my analysis. I used a
single input convolutional neural network, but many other approaches could be tested.
