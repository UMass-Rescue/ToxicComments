# ToxicComments
Implementation to solve the toxic comment classification problem

### Installing the environment
Note: You will need the Anaconda Package Manager to be able to run the code in this repo.
Run the following command from the root directory, to create a conda environment for the code (you can replace toxicenv with any other environment name):

``` conda create --name toxicenv --file spec-file.txt ```

Next, run the following command to activate your newly created environment:

``` conda activate toxicenv ```

Next, run the following command to install the simpletransformers library:

``` pip install simpletransformers ```

You might also need to install apex, which can be done by following the instructions [here](https://github.com/NVIDIA/apex). 

Alternatively, follow the directions here:

- run `cd..` to get out of the root directory
- run `mkdir apex-tools` to create a new folder and then `cd apex-tools` to cd into it
- run `git clone https://github.com/NVIDIA/apex` and then `cd apex`
- now run `pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`
- next, run `mv apex/apex ../ToxicComments/`
- The goal of this process was to install apex, and then move the inner apex folder (yes, there's an apex folder inside the main apex folder) to the root dir, i.e. ToxicComments

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
