import numpy as np
from selfregulation.utils.utils import get_behav_data, get_recent_dataset

dataset = get_recent_dataset()
items = get_behav_data(dataset=dataset, file='items.csv.gz')
subject_items = get_behav_data(dataset=dataset, file='subject_x_items.csv')

# get fmri items alone
# filter for fmri items
grit_items = [
    'New ideas and projects sometimes distract me from previous ones.',
    'Setbacks don\'t discourage me.',
    'I have been obsessed with a certain idea or project for a short time but later lost interest.',
    'I am a hard worker.',
    'I often set a goal but later choose to pursue a different one.',
    'I have difficulty maintaining my focus on projects that take more than a few months to complete.',
    'I finish whatever I begin.',
    'I am diligent.'
]

brief_items = [
    'I am good at resisting temptation.',
    'I have a hard time breaking bad habits.',
    'I am lazy.',
    'I say inappropriate things.',
    'I do certain things that are bad for me, if they are fun.',
    'I refuse things that are bad for me.',
    'I wish I had more self-discipline.',
    'People would say that I have iron self-discipline.',
    'Pleasure and fun sometimes keep me from getting work done.',
    'I have trouble concentrating.',
    'I am able to work effectively toward long-term goals.',
    'Sometimes I can\'t stop myself from doing something, even if I know it is wrong.',
    'I often act without thinking through all the alternatives.'
 ]

future_time_items = [
    'Many opportunities await me in the future.',
    'I expect that I will set many new goals in the future.',
    'My future is filled with possibilities.',
    'Most of my life lies ahead of me.',
    'My future seems infinite to me.',
    'I could do anything I want in the future.',
    'There is plenty of time left in my life to make new plans.',
    'I have the sense that time is running out.',
    'There are only limited possibilities in my future.',
    'As I get older, I begin to experience time as limited.'
 ]


upps_items = [
    "Sometimes when I feel bad, I can't seem to stop what I am doing even though it is making me feel worse.",
    'Others would say I make bad choices when I am extremely happy about something.',
    'When I get really happy about something, I tend to do things that can have bad consequences. ',
    'When overjoyed, I feel like I can’t stop myself from going overboard.',
    'When I am really excited, I tend not to think of the consequences of my actions.',
    'I tend to act without thinking when I am really excited.'
]

impulse_venture_items = [
    'Do you welcome new and exciting experiences and sensations even if they are a little frightening and unconventional?',
    'Do you sometimes like doing things that are a bit frightening?',
    'Would you enjoy the sensation of skiing very fast down a high mountain slope?'
]

fmri_items = grit_items + brief_items + future_time_items + upps_items + impulse_venture_items
fmri_selection = items.query('item_text in %s' % fmri_items)
cols = np.unique([row['survey'] + '.' + str(row['item_num']).zfill(2) for i,row in fmri_selection.iterrows()])

subset = subject_items.loc[:, cols]
