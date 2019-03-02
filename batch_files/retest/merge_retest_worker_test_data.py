import pandas as pd
val_data = pd.read_json('/scratch/users/zenkavi/tmp/data/retest_subs_val_data_post.json')
disc_data = pd.read_json('/scratch/users/zenkavi/tmp/data/retest_subs_disc_data_post.json')
data = pd.concat([disc_data, val_data])
data.reset_index(drop = True, inplace = True)
data.to_json('/scratch/users/zenkavi/tmp/data/retest_subs_test_data_post.json')
