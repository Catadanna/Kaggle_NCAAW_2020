PATH_DIR = "/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/"

FILE_TRAIN_2015 = PATH_DIR + "WEvents2015.csv"
FILE_TRAIN_2016 = PATH_DIR + "WEvents2016.csv"
FILE_TRAIN_2017 = PATH_DIR + "WEvents2017.csv"
FILE_TRAIN_2018 = PATH_DIR + "WEvents2018.csv"
FILE_TRAIN_2019 = PATH_DIR + "WEvents2019.csv"
FILE_TEST = PATH_DIR + "WSampleSubmissionStage1_2020.csv"

NAN_STRING_TO_REPLACE = 'zz'
NAN_VALUE_FLOAT = 8888.0
NAN_VALUE_INT = 8888
NAN_VALUE_STRING = '8888'

BATCH_SIZE = 100
EPOCHS = 5
N_NEURONS = 10

SEED = 8888
SMOOTHING = 0.2

OTHER_NAN = 0
SPLITS = 20

IMPUTING_STRATEGY = 'mean'

PARAMS_ADABOOST = dict()
PARAMS_ADABOOST['n_estimators']=100 
PARAMS_ADABOOST['random_state']=None
PARAMS_ADABOOST['learning_rate']=0.8

PARAMS_CATBOOST = dict()
PARAMS_CATBOOST['logging_level'] = 'Silent'
PARAMS_CATBOOST['eval_metric'] = 'Logloss'
PARAMS_CATBOOST['custom_metric'] = 'Logloss'
PARAMS_CATBOOST['loss_function'] = 'Logloss'
PARAMS_CATBOOST['iterations'] = 40
PARAMS_CATBOOST['od_type'] = 'Iter' # IncToDec, Iter
PARAMS_CATBOOST['random_seed'] = SEED
PARAMS_CATBOOST['learning_rate'] = 0.003 # alpha, default 0.03 if no l2_leaf_reg
PARAMS_CATBOOST['task_type'] = 'CPU'
PARAMS_CATBOOST['use_best_model']: True
PARAMS_CATBOOST['l2_leaf_reg'] = 3.0 # lambda, default 3, S: 300


PARAMS_CATBOOST_REGRESSOR = dict()
PARAMS_CATBOOST_REGRESSOR['logging_level'] = 'Silent'
PARAMS_CATBOOST_REGRESSOR['eval_metric'] = 'RMSE'
PARAMS_CATBOOST_REGRESSOR['custom_metric'] = 'RMSE'
PARAMS_CATBOOST_REGRESSOR['loss_function'] = 'RMSE'
PARAMS_CATBOOST_REGRESSOR['iterations'] = 1
PARAMS_CATBOOST_REGRESSOR['od_type'] = 'Iter' # IncToDec, Iter
#PARAMS_CATBOOST_REGRESSOR['random_seed'] = SEED
PARAMS_CATBOOST_REGRESSOR['learning_rate'] = 0.003 # alpha, default 0.03 if no l2_leaf_reg
PARAMS_CATBOOST_REGRESSOR['task_type'] = 'CPU'
PARAMS_CATBOOST_REGRESSOR['use_best_model']: True
PARAMS_CATBOOST_REGRESSOR['l2_leaf_reg'] = 3.0 # lambda, default 3, S: 300

w_features = [
    'WTeamID', 
    'WFGM', 
    'WFGA', 
    'WFGM3', 
    'WFGA3', 
    'WFTM', 
    'WFTA', 
    'WOR', 
    'WDR', 
    'WAst', 
    'WTO', 
    'WStl', 
    'WBlk', 
    'WPF', 
    'WScore', 
    'Final_WTeam', 
    'Semi_Final_WTeam', 
    'WTeam_W_count', 
    'WScore_mean',
    'WScore_median', 
    'WScore_sum',
    'Diff_WTeam',
    'W_Matches_Tournament',
    'WTeam_Seed',
    #'WTeam_Rank',
    'WTeam_PerCent',
    'WFGA_min', 
    #'WFGA_max', 
    'WFGA_mean', 
    'WFGA_median'
]
l_features = [
    'LTeamID', 
    'LFGM', 
    'LFGA', 
    'LFGM3', 
    'LFGA3', 
    'LFTM', 
    'LFTA', 
    'LOR', 
    'LDR', 
    'LAst', 
    'LTO', 
    'LStl', 
    'LBlk', 
    'LPF', 
    'LScore',
    'Final_LTeam', 
    'Semi_Final_LTeam', 
    'LTeam_L_count', 
    'LScore_mean',  
    'LScore_median', 
    'LScore_sum',
    'Diff_LTeam',
    'L_Matches_Tournament',
    'LTeam_Seed',
    #'LTeam_Rank',
    'LTeam_PerCent',
    'LFGA_min', 
    #'LFGA_max', 
    'LFGA_mean', 
    'LFGA_median'
]

