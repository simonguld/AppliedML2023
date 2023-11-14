import pickle

write, read = True, False

if write:
    dict = {'boosting_type': 'rf', 'min_split_gain': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'objective': 'binary', 'n_jobs': -1, 'n_estimators': 1465, 'learning_rate': 0.33122346404300845, 'max_depth': 26, 'num_leaves': 299, 'bagging_fraction': 0.5516760229916262, 'bagging_freq': 9}


    # save dictionary to person_data.pkl file
    with open('best_params_lgb_rf.pkl', 'wb') as fp:
        pickle.dump(dict, fp)
        print('dictionary saved successfully to file')

if read:
    with open('person_data.pkl', 'rb') as fp:
        person = pickle.load(fp)
        print('Person dictionary')
        print(person)