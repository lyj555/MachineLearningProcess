# -*- coding: utf-8 -*-


RAW_DATA_PATH = ""

pre_result = pre_process(RAW_DATA_PATH)
train_pre, valid_pre, test_pre = split_dataset(pre_result)

train_iter, valid_iter, test_iter = build_iterator(train_pre, valid_pre, test_pre, batch_size=4)

clf = train_by_torch(train_iter, valid_iter, model, optimizer, model_save_path)

pred, label = predict(clf, test_iter)

# pre_un = pre_process(unseen_sample)
# un_iter = build_iterator(pre_un)
# pred, _ = predict(clf, test_iter)


