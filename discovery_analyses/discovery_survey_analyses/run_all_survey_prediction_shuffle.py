"""
make command file to run prediction analyses
"""
import os
import pandas

binary_vars=["Sex","ArrestedChargedLifeCount","DivorceCount","GamblingProblem","ChildrenNumber",
            "CreditCardDebt","RentOwn","RetirementAccount","TrafficTicketsLastYearCount","Obese",
             "TrafficAccidentsLifeCount","CaffienatedSodaCansPerDay","Nervous",
             'Hopeless', 'RestlessFidgety', 'Depressed',
             'EverythingIsEffort', 'Worthless','CigsPerDay','LifetimeSmoke100Cigs',
             'CannabisPast6Months']

with open('run_survey_prediction_shuffle.sh','w') as f:
  for v in binary_vars:
   datafile='/home/01329/poldrack/DATADIR/Self_Regulation_Ontology/discovery_survey_analyses/surveypred/surveypredict_cvresults_%s.csv'%v
   if not os.path.exists(datafile):
        continue
   d=pandas.read_csv(datafile)
   if len(d)==0:
      continue
   if d.testf1[0]<0.55:
        continue
   for i in range(1,251):
    f.write('python demographic_prediction_surveydata.py %s %d\n'%(v,i))
