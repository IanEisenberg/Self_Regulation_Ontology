import numpy as np
import os
import pandas as pd
from selfregulation.utils.utils import  get_info

def gen_reference_item_text(items_df):
    base_directory = get_info('base_directory')
    reference_location = os.path.join(base_directory,'references','variable_name_lookup.csv')
    ref = pd.read_csv(reference_location)
    # add item text
    item_text_lookup = items_df.groupby('item_ID').item_text.unique().apply(lambda x: x[0]).to_dict()
    item_text = [item_text_lookup[i] if i in item_text_lookup.keys() else np.nan for i in ref['Variable Name']]
    # add response text
    response_text_lookup = items_df.groupby('item_ID').response_text.unique().apply(lambda x: ', '.join(x))
    response_text = [response_text_lookup[i].replace('\n','') if i in response_text_lookup.keys() else np.nan for i in ref['Variable Name']]
    ref.loc[:,'Question'] = item_text
    ref.loc[:,'Responses'] = response_text
    ref.to_csv(reference_location, index = False)
