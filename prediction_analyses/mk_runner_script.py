for task in ['regression','classification']:
    for clf in ['lasso','forest']:
        for ds in ['all','survey','task']:
            print('python behav_%s.py 0 %s %s'%(task,clf,ds))
            for i in range(1000):
                print('python behav_%s.py 1 %s %s'%(task,clf,ds))
