import pandas as pd
import requests
import os
import json
import argparse

from ds_utils import create_dir

def parse_CHO(item):
  ID = item['id']
  URI = 'http://data.europeana.eu/item'+ID
  try:
    URL = item['edmIsShownBy'][0]
  except:
    URL = None

  return ID,URI,URL

    
def query_single_category(**kwargs):  
  category = kwargs.get('category')
  skos_concept = kwargs.get('skos_concept')
  n = kwargs.get('n')
  reusability = kwargs.get('reusability','open')

  """

  """

  params = {
      'reusability':reusability,
      'media':True,
      'qf':f'(skos_concept:"{skos_concept}" AND TYPE:IMAGE )', 
      'query':'*', 
      'wskey':'api2demo',
      'sort':'random,europeana_id',
  }

  CHO_list = []
  response = {'nextCursor':'*'}
  while 'nextCursor' in response:
    
    if len(CHO_list)>n:
      break

    params.update({'cursor':response['nextCursor']})
    response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()

    for CHO in response['items']:

      ID,URI,URL = parse_CHO(CHO)

      if URL:
        CHO_list.append({
          'category':category,
          'skos_concept':skos_concept,
          'URI':URI,
          'ID':ID,
          'URL':URL
          })

  return pd.DataFrame(CHO_list[:n])


def main(**kwargs):

  vocab_json = kwargs.get('vocab_json',None)
  n = kwargs.get('n',None)
  name = kwargs.get('name',None)
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

  if name:
    fname = f'{name}.csv'
  else:
    fname = 'dataset.csv'

  df = pd.DataFrame()
  for category,skos_concept in vocab_dict.items():
    print(category)
    df_category = query_single_category(
      category = category,
      skos_concept = skos_concept,
      n = n,
      reusability = reusability,
      )
    df = pd.concat((df,df_category))
    #save after each category
    df.to_csv(os.path.join(saving_dir,fname),index=False)

  return df

  
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
    parser.add_argument('--name', required=False)
    parser.add_argument('--reusability', required=False)
    args = parser.parse_args()

    main(
      vocab_json = args.vocab_json,
      n = args.n,
      name = args.name,
      saving_dir = args.saving_dir,
      reusability = args.reusability

    )
          

    

