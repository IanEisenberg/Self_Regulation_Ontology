import fancyimpute
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, make_scorer

from selfregulation.utils.utils import get_behav_data, get_recent_dataset, get_demographics
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.get_balanced_folds import BalancedKFold

# load data

dataset = get_recent_dataset()
items = get_behav_data(dataset=dataset, file='items.csv.gz')
subject_items = get_behav_data(dataset=dataset, file='subject_x_items.csv')
survey_ontology = load_results(dataset)['survey']
demographics = survey_ontology.DA.data
demo_factors = survey_ontology.DA.get_scores()

# set up prediction
imputer = fancyimpute.KNN()
predictors = imputer.fit_transform(subject_items)
targets = demo_factors.values

# set up cross-validation
for i, name in enumerate(demo_factors.columns):
    CV=BalancedKFold(nfolds=10)
    CV_iter = list(CV.split(predictors, targets[:,0]))
    clf = RidgeCV(cv=5)
    score = cross_val_score(clf,
                            survey_ontology.EFA.get_scores(), 
                            targets[:,i], 
                            cv=CV_iter,
                            scoring=make_scorer(r2_score)).mean()
    print('%s Score: %.2f' % (name, score))


out = survey_ontology.run_prediction(shuffle=False, classifier='ridge')
