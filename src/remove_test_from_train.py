import pandas as pd
import os
import argparse

def remove_test_from_train(**kwargs):

    training_set = kwargs.get('training_set')
    evaluation_set = kwargs.get('evaluation_set')
    saving_dir = kwargs.get('saving_dir')

    training_df = pd.read_csv(training_set)
    eval_df = pd.read_csv(evaluation_set)
    eval_df = eval_df[['URI','ID','URL','category']]
    eval_df = eval_df.dropna()

    training_df = training_df[~training_df.ID.isin(eval_df.ID)]

    training_df.to_csv(os.path.join(saving_dir,'training_data.csv'),index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_set', required=True)
    parser.add_argument('--evaluation_set', required=True)
    parser.add_argument('--saving_dir', required=True)
    args = parser.parse_args()

    remove_test_from_train(training_set = args.training_set, evaluation_set = args.evaluation_set, saving_dir = args.saving_dir)


