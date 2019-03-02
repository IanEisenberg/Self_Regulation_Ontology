import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys

test_data_path = sys.argv[1]
retest_data_path = sys.argv[2]
t = sys.argv[3]

def calc_shift_DV(df, dvs = {}):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # subset df
    df = df.query('rt != -1').reset_index(drop = True)

    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean()}
    dvs['avg_rt'] = {'value':  df.rt.median()}

    try:
        rs = smf.glm('correct ~ trials_since_switch+trial_num', data = df, family = sm.families.Binomial()).fit()
        learning_rate = rs.params['trials_since_switch']
        learning_to_learn = rs.params['trial_num']
    except:
        learning_rate = 'NA'
        learning_to_learn = 'NA'

    dvs['learning_to_learn'] = {'value': learning_to_learn}
    dvs['learning_rate'] = {'value':  learning_rate}

    #conceptual_responses: The CLR score is the total number of consecutive correct responses in a sequence of 3 or more.
    CLR_score = 0
    CLR_thresh_sum= 0 # must be >= 3
    #fail_to_maintain_set: The FTMS score is the number of sequences of 5 correct responses or more,
    #followed by an error, before attaining the 10 necessary for a set change - for us just counting number of streaks of >5 since 10 isn't necessary for set change
    FTMS_score = 0
    FTMS_thresh_sum = 0

    for i, row in df.iterrows():
        if row.correct==True:
            CLR_thresh_sum += 1
            FTMS_thresh_sum += 1
        else:
            if FTMS_thresh_sum >= 5:
                FTMS_score += 1
            CLR_thresh_sum = 0
            FTMS_thresh_sum = 0
        if CLR_thresh_sum>=3:
            CLR_score+=1

    dvs['conceptual_responses'] = {'value': CLR_score}
    dvs['fail_to_maintain_set'] = {'value': FTMS_score}


    #add last_rewarded_feature column by switching the variable to the feature in the row right before a switch and assigning to the column until there is another switch
    df['last_rewarded_feature'] = "NaN"
    last_rewarded_feature = "NaN"
    for i, row in df.iterrows():
        if row.shift_type != 'stay':
            last_rewarded_feature = df.rewarded_feature.iloc[i-1]
        df.last_rewarded_feature.iloc[i] = last_rewarded_feature

    #perseverative_responses: length of df where the choice_stim includes the last_rewarded_feature
    try:
        perseverative_responses = df[df.apply(lambda row: row.last_rewarded_feature in str(row.choice_stim), axis=1)]
    except:
        perseverative_responses = pandas.DataFrame(columns=['correct'])
    dvs['perseverative_responses'] = {'value': len(perseverative_responses)}
    #perseverative_errors: length of perseverative_responses df that is subsetted by incorrect responses
    dvs['perseverative_errors'] = {'value': len(perseverative_responses.query("correct == 0"))}
    #total_errors
    dvs['total_errors'] = {'value': len(df.query("correct==0"))}
    #nonperseverative_errors
    dvs['nonperseverative_errors'] = {'value': len(df.query("correct==0")) - dvs['perseverative_errors']['value']}

    return dvs

#wrapper function to calculate dvs for each break
def calc_breaks(df):
    df = df.query('exp_stage != "practice"')
    num_trials = df.shape[0]
    breaks = numpy.arange(10, num_trials, 10)
    out_df = pandas.DataFrame()
    print(df.worker_id.unique())
    for b in breaks:
        tmp_df = df.iloc[:b]
        print(max(tmp_df.trial_num))
        tmp_dvs = pandas.DataFrame.from_dict(calc_shift_DV(tmp_df))
        out_df = out_df.append(tmp_dvs, ignore_index = True)
    return(out_df)

#load requested raw data
if t == "t1":
    data = pandas.read_csv(test_data_path+'Individual_Measures/shift_task.csv.gz', compression='gzip')
elif t == "t2":
    data = pandas.read_csv(retest_data_path+'Individual_Measures/shift_task.csv.gz', compression='gzip')

#extract retest workers from raw data
retest_workers = ['s198', 's409', 's473', 's286', 's017', 's092', 's403', 's103','s081', 's357', 's291', 's492', 's294', 's145', 's187', 's226','s368', 's425', 's094', 's430', 's376', 's284', 's421', 's034','s233', 's027', 's108', 's089', 's196', 's066', 's374', 's007','s509', 's365', 's305', 's453', 's504', 's161', 's441', 's205','s112', 's218', 's129', 's093', 's180', 's128', 's170', 's510','s502', 's477', 's551', 's307', 's556', 's121', 's237', 's481','s259', 's467', 's163', 's111', 's427', 's508', 's190', 's091','s207', 's484', 's449', 's049', 's336', 's212', 's142', 's313','s369', 's165', 's028', 's216', 's346', 's083', 's391', 's388','s384', 's275', 's442', 's505', 's098', 's456', 's209', 's372','s179', 's168', 's084', 's329', 's373', 's065', 's277', 's026','s011', 's063', 's507', 's005', 's495', 's501', 's032', 's326','s396', 's420', 's469', 's244', 's359', 's110', 's383', 's254','s060', 's339', 's380', 's471', 's206', 's182', 's500', 's314','s285', 's086', 's012', 's097', 's149', 's192', 's173', 's262','s273', 's402', 's015', 's014', 's085', 's489', 's071', 's062','s042', 's009', 's408', 's184', 's106', 's397', 's451', 's269','s295', 's265', 's301', 's082', 's238', 's328', 's334']
data = data[data['worker_id'].isin(retest_workers)]
data = data.reset_index()

#groupby subjects and apply the function that would calculate the dv's for each break
out = data.groupby('worker_id').apply(calc_breaks)
out = out.reset_index(level=['worker_id', None])

#write out output
out.to_csv(retest_data_path+ 'Local/'+t+'_shift_dvs.csv')