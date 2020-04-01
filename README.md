# ToxicComments
Implementation to solve the toxic comment classification problem

### Installing the environment
Note: You will need the Anaconda Package Manager to be able to run the code in this repo.
Run the following command from the root directory, to create a conda environment for the code (you can replace myenv with any other environment name):

``` conda create --name myenv --file spec-file.txt ```

Alternatively, to update an existing conda environment, run the following command:

``` conda update --name myenv --file spec-file.txt ```

Next, run the following command to activate your newly created environment:

``` conda activate myenv ```

You might also need to install apex, which can be done by following the instructions [here](https://github.com/NVIDIA/apex) (try this if something breaks)

You are now ready to run toxicity.py!

### Toxicity.py
This file attempts to solve the Toxic Comment Challenge from Kaggle, which is essentially a multi-label classification problem. To run the file on the Kaggle dataset, as-is, simply run the following command:
```  python toxicity.py -l labels.txt -d ./data ```

# Running Toxicity.py on your own dataset
If you want to run toxicity.py on your own data, you need to take the following steps:

### Lables.txt
This file contains all of your labels, separated by newlines. It is important to note that the labels in this file follow the same order as the columns in the dataset. Replace the labels with your own, and you should be good to go.

### Data
Make sure the data is in csv format, and that its structure is identical to the default dataset. The number of columns/labels don't matter. 

And that's it! Happy classifying :)
