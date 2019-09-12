# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import shap

# print the JS visualization code to the notebook
# shap.initjs()

X,y = shap.datasets.adult()
# X_display,y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)


params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

pre = model.predict(X)

shap_values[1].shape

clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)

clf = xgb.XGBClassifier(**{"learning_rate": 0.01, "num_boost_round": 30})
clf.fit(X_train, y_train)

explainer = shap.TreeExplainer(clf)
shap_values1 = explainer.shap_values(X)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)


explainer = shap.TreeExplainer(clf)
shap_values1 = explainer.shap_values(X_test)