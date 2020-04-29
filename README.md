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

You can also change the hyperparameters by passing in a `dict` containing the relevant attributes to the `train_model` method.

The train_model method will create a checkpoint (save) of the model at every nth step where n is `self.args['save_steps']`. Upon completion of training, the final model will be saved to `self.args['output_dir']`.

## Running Toxicity.py on your own dataset
If you want to run toxicity.py on your own data, you need to take the following steps:

### Lables.txt
This file contains all of your labels, separated by newlines. It is important to note that the labels in this file follow the same order as the columns in the dataset. Replace the labels with your own, and you should be good to go.

### Data
Make sure the data is in csv format, and that its structure is identical to the default dataset. The number of columns/labels don't matter. 

And that's it! Happy classifying :)

## Hardware
This code was run on 4 NVIDIA GeForce GTX 1080 Ti GPUs, as well as Dan's machine (by Jagath).

## Benchmarks
NOTE: These results are relevant to the current dataset, but your mileage may vary with your own data.

Submitting to kaggle nets a score of 0.98, where the score is the mean area under ROC curve.

On 4 NVIDIA GeForce GTX 1080 Ti GPUs, 1 epoch on the training data takes 1.5 hours to finish.

