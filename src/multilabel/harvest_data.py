import pandas as pd
import requests
import os
import json
import argparse
from tqdm import tqdm
from random import choice
from itertools import combinations

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
import sys
sys.path.append(os.path.join(ROOT_DIR))
from ds_utils import create_dir
from harvest_data_single_label import parse_CHO

def combine_categories(vocab,min_n_categories,max_n_categories):
  categories_list = list(vocab.keys())

  combinations_list = []
  for n in range(min_n_categories,max_n_categories+1):
    combinations_list += list(combinations(categories_list, n))

  return combinations_list


def search_combination(concepts_list,vocab,reusability):

  skos_concepts_list = [vocab[concept] for concept in concepts_list]

  query = '"'+'"AND"'.join(skos_concepts_list)+'"'
  params = { 
      'reusability':reusability,
      'media':True,
      'qf':f'TYPE:IMAGE',
      'query':query,
      'wskey':'api2demo',
      'sort':'random,europeana_id',
      }

  response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()
  return response

def search_number_combinations(combinations_list,vocab,reusability):

  results = []
  for comb in tqdm(combinations_list):
    response = search_combination(comb,vocab,reusability)
    n_res = response['totalResults']
    if n_res>0:
      results.append({'labels':' '.join(comb),'results':n_res})

  return pd.DataFrame(results)


def search_CHOs(concepts_list,vocab,reusability,N):

  skos_concepts_list = [vocab[concept] for concept in concepts_list]
  query = '"'+'"AND"'.join(skos_concepts_list)+'"'
  params = { 
      'reusability':reusability,
      'media':True,
      'qf':f'TYPE:IMAGE',
      'query':query,
      'wskey':'api2demo',
      'sort':'random,europeana_id',
      }

  CHO_list = []
  response = {'nextCursor':'*'}
  while 'nextCursor' in response:
    params.update({'cursor':response['nextCursor']})

    response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()      
    CHO_list += response['items']
    if len(CHO_list)>N:
      break

  return CHO_list[:N]

def assemble_multilabel_dataset(df,vocab,reusability,N):
  CHO_list = []
  for row in tqdm(list(df.iterrows())):
    concept_list = row[1]['labels'].split()
    retrieved_CHO_list = search_CHOs(concept_list,vocab,reusability,N)
    skos_concepts_list = [vocab[concept] for concept in concept_list]

    for CHO in retrieved_CHO_list:
      ID,URI,URL = parse_CHO(CHO)
      if URL:
        CHO_list.append({
          'category':' '.join(concept_list),
          'skos_concept':' '.join(skos_concepts_list),
          'URI':URI,
          'ID':ID,
          'URL':URL
          })
        
  return pd.DataFrame(CHO_list)

def remove_duplicates(df):
  #filter repeated images accross different combinations of concepts
  #order the dataframe in decreasing order of number of labels and keep first occurrence
  _df = df.copy()
  _df['n_labels'] = _df['category'].apply(lambda x: len(x.split()))
  _df = _df.sort_values(by='n_labels',ascending=False)
  _df = _df.drop_duplicates(subset='ID',keep='first')
  return _df[['category','skos_concept','ID','URI','URL']]


def main(**kwargs):

    vocab_json = kwargs.get('vocab_json',None)
    n = kwargs.get('n',None)
    saving_dir = kwargs.get('saving_dir',None)
    reusability = kwargs.get('reusability','open')

    if not vocab_json:
        raise Exception('vocab_json not provided')
    if not saving_dir:
        raise Exception('saving_dir not provided')

    if not n:
        n = 3000
    else:
        n = int(n)

    with open(vocab_json,'r') as f:
        vocab_dict = json.load(f)



    create_dir(saving_dir)

    min_n_categories = 2
    max_n_categories = 3
    combinations_list = combine_categories(vocab_dict,min_n_categories,max_n_categories)

    combination_df = search_number_combinations(combinations_list,vocab_dict,reusability)
    combination_df.to_csv(os.path.join(saving_dir,f'combinations_{reusability}.csv'),index=False)

    df = assemble_multilabel_dataset(combination_df,vocab_dict,reusability,n)
    df.to_csv(os.path.join(saving_dir,f'multilabel_dataset_{reusability}.csv'),index=False)



  
if __name__ == '__main__':

    """
    Script for assembling the image classification dataset 
    in csv format making use of Europeana's Search API 

    Usage:

      python src/harvest_data.py --vocab_json vocabulary.json --n 3000 --name dataset_3000

    Parameters:

      vocab_json: json file with categories as keys and concept URIs as values
                  Required

      saving_dir: directory for saving the csv file. 
                  If not specified this will be the root path of the repository

      name: tag for the table
             Default: dataset
      
      n: number of desired Cultural Heritage Objects per category
         Default: 1000

      reusability: level of copyright of the CHOs. Available: open, permission, restricted
         Default: open
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_json', required=True)
    parser.add_argument('--saving_dir', required=True)
    parser.add_argument('--n', required=False, nargs = '?', const = 1000)
    parser.add_argument('--reusability', required=False)
    args = parser.parse_args()

    main(
      vocab_json = args.vocab_json,
      n = args.n,
      saving_dir = args.saving_dir,
      reusability = args.reusability

    )
          

    

