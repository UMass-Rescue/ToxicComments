import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import MultiLabelClassificationModel
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--labels", type=str,
	help="path to labels.txt")
ap.add_argument("-d", "--data", type=str,
	help="path to data directory")

args = vars(ap.parse_args())

labels_file = args['labels']
with open(labels_file, 'r') as f:
    labels_values = [x.strip() for x in f.readlines()]

label_count = len(labels_values)

data_dir = args["data"]
if data_dir[-1] != '/':
	data_dir += '/'

df = pd.read_csv(data_dir + 'train.csv')

text = df.iloc[:, 1]
labels = df.iloc[:, 2:]

text.reset_index(drop=True, inplace=True)
labels.reset_index(drop=True, inplace=True)

labels_list = labels.values.tolist()
labels_list = [[x] for x in labels_list]
labels = pd.DataFrame(labels_list, columns=['labels'])

train_temp = pd.concat([text, labels], axis=1)
train_data = train_temp.rename({'comment_text': 'text'}, axis=1)

train_df, eval_df = train_test_split(train_data, test_size=0.2)

model = MultiLabelClassificationModel('roberta', 
                                    'roberta-base', 
                                    num_labels=label_count,  
                                    args={'train_batch_size':2, 
                                        'gradient_accumulation_steps':16, 
                                        'learning_rate': 3e-5, 
                                        'num_train_epochs': 3, 
                                        'max_seq_length': 256,
                                        'reprocess_input_data': True})


model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

test_df = pd.read_csv(data_dir + 'test.csv')

to_predict = test_df.comment_text.apply(lambda x: x.replace('\n', ' ')).tolist()
preds, outputs = model.predict(to_predict)

sub_df = pd.DataFrame(outputs, columns=labels_values) 


sub_df['id'] = test_df['id']
sub_df = sub_df[['id'] + labels_values] 

sub_df.to_csv('outputs/submission.csv', index=False)