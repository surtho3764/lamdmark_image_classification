# library
import os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold



# map landmark_id to label code ,split K-fold ,get data info function
# 將landmark_id轉化成label值，按照數字編號從0開始label值，並把對應表儲存成pickle檔案
def convert_landmark_id_label_code(data_df):
    df_train = data_df
    df_train = df_train.drop(columns=['url'])
    cls = df_train['landmark_id'].unique()
    landmark_id_idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}

    with open('idx2landmark_id.pkl', 'wb') as fp:
        pickle.dump(landmark_id_idx, fp)
    return landmark_id_idx


# split K-fold
# 將資料做K-fold，並儲存在'fold'欄位
def split_k_fold(data_df):
    df_train = data_df
    df_train['fold'] = -1

    # split K-fold
    skf = StratifiedKFold(2, shuffle=True, random_state=32)
    for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['landmark_id'])):
        df_train.loc[valid_idx, 'fold'] = i

    # df_train.to_csv('train_0.csv',index=False)
    return df_train


# get data info
# 得到影像標籤和影像位置表，之後用來讀取影像的位置資訊
def get_df(data_dir, file_name):
    train_file_dir = data_dir + '/' + file_name
    df_train = pd.read_csv(train_file_dir)

    # map landmark_id to label code
    landmark_id_idx = convert_landmark_id_label_code(df_train)
    df_train['ori_landmark_id'] = df_train['landmark_id']
    df_train['landmark_id'] = df_train['landmark_id'].map(landmark_id_idx)

    # split K-fold
    df_train = split_k_fold(df_train)

    # filepath
    df_train['filepath'] = df_train['id'].apply(lambda x: os.path.join(data_dir, 'train', x + '.jpg'))
    # df_train.to_csv('train_info.csv',index=False)

    # get label
    lable = df_train['landmark_id'].unique()
    # get label numner
    out_dim = df_train['landmark_id'].nunique()
    return df_train, out_dim
