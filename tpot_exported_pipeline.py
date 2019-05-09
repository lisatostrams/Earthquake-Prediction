import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-7.05345845004224
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=17, min_samples_split=8, n_estimators=100)),
            StackingEstimator(estimator=AdaBoostRegressor(learning_rate=1.0, loss="exponential", n_estimators=100)),
            Nystroem(gamma=0.1, kernel="additive_chi2", n_components=7)
        ),
        FunctionTransformer(copy)
    ),
    ElasticNetCV(l1_ratio=0.9, tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
