# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split



def random_search(train_x, train_y, k_fold=None, create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, 
                  sample_ratio=0.8, max_iter=10, model="rf", random_state=None,
                  effect_threshold=None):
    """
    :param 
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    best_effect, best_subset, best_feat_dim = float("-inf"), "", train_x.shape[1]
    t = 0
    
    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, train_y, valid_x, valid_y = train_test_split(train_x, train_y, valid_ratio)
    
    while t < max_iter:
        feature_subset = generate_random_list(X.columns, sample_ratio)
        feature_dim = len(feature_subset)
        
        if k_fold is not None:
            effect_subset = cross_validation(train_x[feature_subset], train_y, k_fold, model=model)
        else:
            effect_subset = valid_score(train_x[feature_subset], train_y, valid_x[featuer_subset], valid_y, model=model)
        
        best_effect, best_subset, feature_dim = update_effect(old_effect=(best_effect, best_subset, best_feat_dim), 
                                                              now_effect=(effect_subset, feature_subset, feature_dim))
        t += 1
        # if effect_threshold is None:
        #    if effect_subset > best_effect:
        #        best_effect, best_subset, feature_dimension = effect_subset, feature_subset, feature_dimension
        #    elif effect_subset == best_effect:
        #        if feature_dimension
    return best_subset.split(",")
        
    