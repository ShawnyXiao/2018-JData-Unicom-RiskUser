# coding: utf-8

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
import time
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')


# Constants define
ROOT_PATH = '../'
ONLINE = 1

target = 'label'
train_len = 4999
threshold = 0.5


########################################### Helper function ###########################################


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

def merge_feat_count(df, df_feat, columns_groupby, new_column_name, type='int'):
    df_count = pd.DataFrame(df_feat.groupby(columns_groupby).size()).fillna(0).astype(type).reset_index()
    df_count.columns = columns_groupby + [new_column_name]
    df = df.merge(df_count, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_onehot_count(df, df_feat, columns_groupby, prefix, type='int'):
    df_count = df_feat.groupby(columns_groupby).size().unstack().fillna(0).astype(type).reset_index()
    df_count.columns = [i if i == columns_groupby[0] else prefix + '_' + str(i) for i in df_count.columns]
    df = df.merge(df_count, on=columns_groupby[0], how='left')
    return df, list(np.delete(df_count.columns.values, 0))

def merge_feat_nunique(df, df_feat, columns_groupby, column, new_column_name, type='int'):
    df_nunique = pd.DataFrame(df_feat.groupby(columns_groupby)[column].nunique()).fillna(0).astype(type).reset_index()
    df_nunique.columns = columns_groupby + [new_column_name]
    df = df.merge(df_nunique, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_min(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_min = pd.DataFrame(df_feat.groupby(columns_groupby)[column].min()).fillna(0).astype(type).reset_index()
    df_min.columns = columns_groupby + [new_column_name]
    df = df.merge(df_min, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_max(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_max = pd.DataFrame(df_feat.groupby(columns_groupby)[column].max()).fillna(0).astype(type).reset_index()
    df_max.columns = columns_groupby + [new_column_name]
    df = df.merge(df_max, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_mean(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_mean = pd.DataFrame(df_feat.groupby(columns_groupby)[column].mean()).fillna(0).astype(type).reset_index()
    df_mean.columns = columns_groupby + [new_column_name]
    df = df.merge(df_mean, on=columns_groupby, how='left')
    return df, [new_column_name]

def eval_auc_f1(preds, dtrain):
    df = pd.DataFrame({'y_true': dtrain.get_label(), 'y_score': preds})
    df['y_pred'] = df['y_score'].apply(lambda x: 1 if x >= threshold else 0)
    auc = metrics.roc_auc_score(df.y_true, df.y_score)
    f1 = metrics.f1_score(df.y_true, df.y_pred)
    return 'feval', (auc * 0.6 + f1 * 0.4), True

def lgb_cv(train_x, train_y, params, rounds, folds):
    start = time.clock()
    log(str(train_x.columns))
    dtrain = lgb.Dataset(train_x, label=train_y)
    log('run cv: ' + 'round: ' + str(rounds))
    res = lgb.cv(params, dtrain, rounds, nfold=folds, 
                 metrics=['eval_auc_f1', 'auc'], feval=eval_auc_f1, 
                 early_stopping_rounds=200, verbose_eval=5)
    elapsed = (time.clock() - start)
    log('Time used:' + str(elapsed) + 's')
    return len(res['feval-mean']), res['feval-mean'][len(res['feval-mean']) - 1], res['auc-mean'][len(res['auc-mean']) - 1]

def lgb_train_predict(train_x, train_y, test_x, params, rounds):
    start = time.clock()
    log(str(train_x.columns))
    dtrain = lgb.Dataset(train_x, label=train_y)
    valid_sets = [dtrain]
    model = lgb.train(params, dtrain, rounds, valid_sets, feval=eval_auc_f1, verbose_eval=5)
    pred = model.predict(test_x)
    elapsed = (time.clock() - start)
    log('Time used:' + str(elapsed) + 's')
    return model, pred

def store_result(test_index, pred, threshold, name):
    result = pd.DataFrame({'uid': test_index, 'prob': pred})
    result = result.sort_values('prob', ascending=False)
    result['label'] = 0
    result.loc[result.prob > threshold, 'label'] = 1
    result.to_csv('../data/output/sub/' + name + '.csv', index=0, header=0, columns=['uid', 'label'])
    return result


########################################### Read data ###########################################


train = pd.read_csv(ROOT_PATH + 'data/input/train/uid_train.txt', header=None, sep='\t')
train.columns = ['uid', 'label']
train_voice = pd.read_csv(ROOT_PATH + 'data/input/train/voice_train.txt', header=None, sep='\t')
train_voice.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
train_sms = pd.read_csv(ROOT_PATH + 'data/input/train/sms_train.txt', header=None, sep='\t')
train_sms.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']
train_wa = pd.read_csv(ROOT_PATH + 'data/input/train/wa_train.txt', header=None, sep='\t')
train_wa.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date']

test = pd.DataFrame({'uid': ['u' + str(i) for i in range(5000, 7000)]})
test_voice = pd.read_csv(ROOT_PATH + 'data/input/test_a/voice_test_a.txt', header=None, sep='\t')
test_voice.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
test_sms = pd.read_csv(ROOT_PATH + 'data/input/test_a/sms_test_a.txt', header=None, sep='\t')
test_sms.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']
test_wa = pd.read_csv(ROOT_PATH + 'data/input/test_a/wa_test_a.txt', header=None, sep='\t')
test_wa.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date']

df = pd.concat([train, test]).reset_index(drop=True)
df_voice = pd.concat([train_voice, test_voice]).reset_index(drop=True)
df_sms = pd.concat([train_sms, test_sms]).reset_index(drop=True)
df_wa = pd.concat([train_wa, test_wa]).reset_index(drop=True)


########################################### Feature engineer ###########################################


predictors = []

df, predictors_tmp = merge_feat_count(df, df_voice, ['uid'], 'count_gb_uid_in_voice'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_count(df, df_sms, ['uid'], 'count_gb_uid_in_sms'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_count(df, df_wa, ['uid'], 'count_gb_uid_in_wa'); predictors += predictors_tmp

df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'opp_len'], 'voice_opp_len'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'call_type'], 'voice_call_type'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'in_out'], 'voice_in_out_'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'opp_len'], 'sms_opp_len'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'in_out'], 'sms_in_out'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_wa, ['uid', 'wa_type'], 'wa_type'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_wa, ['uid', 'date'], 'wa_date'); predictors += predictors_tmp

df, predictors_tmp = merge_feat_nunique(df, df_voice, ['uid'], 'opp_num', 'nunique_oppNum_gb_uid_in_voice'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_voice, ['uid'], 'opp_head', 'nunique_oppHead_gb_uid_in_voice'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_sms, ['uid'], 'opp_num', 'nunique_oppNum_gb_uid_in_sms'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_sms, ['uid'], 'opp_head', 'nunique_oppHead_gb_uid_in_sms'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_wa, ['uid'], 'wa_name', 'nunique_waName_gb_uid_in_wa'); predictors += predictors_tmp

col_list = ['visit_cnt', 'visit_dura', 'up_flow', 'down_flow']
for i in col_list:
    df, predictors_tmp = merge_feat_min(df, df_wa, ['uid'], i, 'min_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_wa, ['uid'], i, 'max_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_wa, ['uid'], i, 'mean_%s_gb_uid_in_wa' % i); predictors += predictors_tmp

train_x = df.loc[:(train_len - 1), predictors]
train_y = df.loc[:(train_len - 1), target]
test_x = df.loc[train_len:, predictors]


########################################### LightGBM ###########################################


config_lgb = {
    'rounds': 10000,
    'folds': 5
}

params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 63,
    'learning_rate': 0.06,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    # 'min_sum_hessian_in_leaf': 10,
    'verbosity': 5,
    'num_threads': cpu_count() - 1,
    'seed': 7,
}

# lgb_cv(train_x, train_y, params_lgb, config_lgb['rounds'], config_lgb['folds'])

model_lgb, pred_lgb = lgb_train_predict(train_x, train_y, test_x, params_lgb, 90)

result = store_result(test.uid, pred_lgb, threshold, '20180523-lgb-%d-%d(r%d)' % (7742, 9098, 90))
result = store_result(test.uid, pred_lgb, threshold, 'submission')

imp = pd.DataFrame({'feature':train_x.columns.values, 'importance':list(model_lgb.feature_importance())})
imp = imp.sort_values(by = 'importance', ascending = False)
imp.to_csv(ROOT_PATH + 'data/output/feat_imp/imp-20180523-%d-%d(r%d).csv' % (7700, 9102, 90), index=False)
