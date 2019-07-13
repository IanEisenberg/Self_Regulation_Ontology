"""
utilities for fixing metadata
"""
import os,json,sys

from selfregulation.utils.utils import get_info

#warnings.filterwarnings("ignore") # only turn this on in production mode
                                  # to keep log files from overflowing

nruns=10
dataset='Complete_10-27-2017'
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Complete_Data/%s'%dataset)


outdir=os.path.join(derived_dir,'metadata')

def write_metadata(metadata,fname,
    outdir=outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with open(os.path.join(outdir,fname), 'w') as outfile:
            json.dump(metadata, outfile, sort_keys = True, indent = 4,
                  ensure_ascii=False)
    return outdir
def load_metadata(variable,
    outdir=outdir):

    with open(os.path.join(outdir,'%s.json'%variable)) as outfile:
            metadata=json.load(outfile)
    return metadata

def metadata_subtract_one(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        NewLevels['%d'%int(int(l)-1)]=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md

def metadata_reverse_scale(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        NewLevels['%d'%int(int(l)*-1 + len(LevelsOrig))]=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md

def metadata_replace_zero_with_nan(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        if not l.find('0')>-1:
            NewLevels[l]=LevelsOrig[l]
        else:
            NewLevels['n/a']=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md

def metadata_change_two_to_zero_for_no(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        if l.find('2')>-1:
            NewLevels['0']=LevelsOrig[l]
        else:
            NewLevels[l]=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md
