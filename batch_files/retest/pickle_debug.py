"""
Pickle backend module

Store the trace in a pickle file.

Notes
-----
Pickle file are not always compatible across different python
versions. Users should use this backend only for shortlived projects.

"""

from . import ram, no_trace, base
import os
import datetime
import numpy
import string

try:
    import cPickle as std_pickle
except ImportError:
    import pickle as std_pickle   # In Python 3, cPickle is folded into pickle

from pymc import six

__all__ = ['Trace', 'Database', 'load']


class Trace(ram.Trace):
    pass


class Database(base.Database):
    def __init__(self, dbname=None, dbmode='a'):
        """Assign a name to the file the database will be saved in.

        :Parameters:
        dbname : string
          Name of the pickle file.
        dbmode : {'a', 'w'}
          File mode.  Use `a` to append values, and `w` to overwrite
          an existing file.
        """
        self.__name__ = 'pickle'
        self.filename = dbname
        self.__Trace__ = Trace
        self.trace_names = []
        # A list of sequences of names of the objects to tally.
        self._traces = {}  # A dictionary of the Trace objects.
        self.chains = 0
        if os.path.exists(dbname):
            if dbmode == 'w':
                os.remove(dbname)
    def _finalize(self):
        """Dump traces using cPickle."""
        container = {}
        try:
            for name in self._traces:
                container[name] = self._traces[name]._trace
            container['_state_'] = self._state_
            file = open(self.filename, 'w+b')
            std_pickle.dump(container, file)
            file.close()
        except AttributeError:
            pass


import sys
from os import path
sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/batch_files/retest')
from get_dbpath import get_dbpath

def load(filename):
    try:
        file = open(filename, 'rb')
    except FileNotFoundError:
        db_name = filename.split('/')[-1]
        db_path = get_dbpath()
        file = open(path.join(db_path, db_name), 'rb')
    container = std_pickle.load(file)
    file.close()
    db = Database(file.name)        
    chains = 0
    funs = set()
    for k, v in six.iteritems(container):
        if k == '_state_':
            db._state_ = v
        else:
            db._traces[k] = Trace(name=k, value=v, db=db)
            setattr(db, k, db._traces[k])
            chains = max(chains, len(v))
            funs.add(k)

    db.chains = chains
    db.trace_names = chains * [list(funs)]

    return db
