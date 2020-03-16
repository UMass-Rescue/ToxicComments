import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import MultiLabelClassificationModel


df = pd.read_csv('data/train.csv')

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
                                    num_labels=6,  # HARDCODED
                                    args={'train_batch_size':2, 
                                        'gradient_accumulation_steps':16, 
                                        'learning_rate': 3e-5, 
                                        'num_train_epochs': 3, 
                                        'max_seq_length': 256,
                                        'reprocess_input_data': True})


model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

test_df = pd.read_csv('data/test.csv')

to_predict = test_df.comment_text.apply(lambda x: x.replace('\n', ' ')).tolist()
preds, outputs = model.predict(to_predict)

sub_df = pd.DataFrame(outputs, columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']) # HARDCODED


sub_df['id'] = test_df['id']
sub_df = sub_df[['id', 'toxic','severe_toxic','obscene','threat','insult','identity_hate']] # HARDCODED

sub_df.to_csv('outputs/submission.csv', index=False)