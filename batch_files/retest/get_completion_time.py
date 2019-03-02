import pandas as pd
data = pd.read_json('/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_complete_data_post.pkl')
completion_dates = data.groupby(['worker_id'], sort=False)['finishtime'].max()
completion_dates.to_csv('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_02-03-2018/Local/complete_completion_dates.csv')
