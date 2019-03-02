#!/usr/bin/env python

import sys
import pickle

d=pickle.load(open(sys.argv[1],'rb'))
print(d)
