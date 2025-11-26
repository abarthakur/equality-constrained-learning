import subprocess
import sys
import os

result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True)
gitroot = result.stdout.strip()
sys.path.append(gitroot)
# sys.path.append(os.path.join(gitroot, "external"))

from src.utils import get_git_root

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
import numpy as np
import torch
import pickle
import argparse


class CompasA : 
    '''
    Container class for utility functions. No real state, except hardcoded values. Main interface is load_split_processed.
    '''
    
    def __init__(self, protected_col ='race'):
        gitroot = get_git_root()
        self.data_dir =  os.path.join(gitroot, 'data/fairness/compas-analysis')
        # assert that the data_dir exists
        assert os.path.exists(self.data_dir), f"{self.data_dir} does not exist"
        # then create data_dir/processed if it does not exist
        processed_dir = os.path.join(self.data_dir, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        self.keep_cols = ['sex', 'age', 'race', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid']
        self.categorical_cols=['race']
        self.binary_cols = ['sex', 'c_charge_degree']
        self.bin_cols=['priors_count', 'juv_misd_count', 'juv_other_count']
        self.bin_cols_bins=[[0,0.99,1,2,3,4,1000], [0,0.99,1,1000], [0,0.99,1,1000]]
        self.special_cols=['age', 'two_year_recid']
        assert len(self.categorical_cols + self.binary_cols +self.bin_cols+self.special_cols)==len(self.keep_cols)
        assert protected_col in ['sex', 'race']
        self.protected_col=protected_col

    def load_partial_processed(self):
        df=pd.read_csv(f"{self.data_dir}/compas-scores-two-years.csv", sep=",")
        # filter rows
        df=df[(df['days_b_screening_arrest'] >= -30) & (df['days_b_screening_arrest'] <= 30)]
        # filter cols
        df=df[self.keep_cols]
        df['race'] = df['race'].replace({'Asian':'Other', 'Native American':'Other'})
        return df

    def get_categorical_encoders(self, full_df) :
        encoders={}
        encoders['cat']=OneHotEncoder(sparse_output=False, dtype=np.float64).fit(full_df[self.categorical_cols])
        encoders['binary']=OneHotEncoder(sparse_output=False, dtype=np.float64, drop="first").fit(full_df[self.binary_cols])
        encoders['protected']=OrdinalEncoder(dtype=np.float64).fit(full_df[[self.protected_col]])
        return encoders 
    
    def process_dataset(self, df, train, encoders, return_type):
        df_final=[]
            
        # categorical columns 
        cat_encoded = encoders["cat"].transform(df[self.categorical_cols])
        cat_columns = [f"{col}_one_hot_{name}" for col, vals in zip(self.categorical_cols, encoders["cat"].categories_) 
                       for name in vals]
        df_final += [pd.DataFrame(cat_encoded, columns=cat_columns)]
        
        # binary columns 
        bin_encoded = encoders["binary"].transform(df[self.binary_cols])
        bin_columns = [f"{col}_one_hot_{name}" for col, vals in zip(self.binary_cols, encoders["binary"].categories_) for i, name in enumerate(vals) if i==1]
        df_final += [pd.DataFrame(bin_encoded, columns=bin_columns)]
        
        # columns to bin
        for i,col in enumerate(self.bin_cols): 
            binned_df = pd.get_dummies(pd.cut(df[col], bins =  self.bin_cols_bins[i], include_lowest=True), prefix=col, dtype=np.float64)
            df_final +=[binned_df]
        
        # age based on quantiles
        if train:
            encoders["age"] = KBinsDiscretizer(n_bins=5, strategy="quantile", encode="onehot-dense").fit(df[["age"]])
        age_encoded = encoders["age"].transform(df[["age"]])
        age_cols = [f"age_quantile_{i}" for i in range(0, encoders["age"].n_bins_[0])]
        df_final += [pd.DataFrame(age_encoded, columns=age_cols)] 
        
        # target column : two_year_recid
        target_encoded = (df['two_year_recid']==1).values*1.0
        df_final+=[pd.DataFrame(target_encoded, columns=["target"])]
        
        # add protected column 
        protected_col=encoders["protected"].transform(df[[self.protected_col]]).flatten()
        df_final+=[pd.DataFrame(protected_col, columns=[f"protected_{self.protected_col}"])]
        
        # concatenate dataframes   
        # reset index, otherwise mixture of hand crafted and processed dataframes will mess things up
        df_final = [part.reset_index(drop=True) for part in df_final]        
        df_final = pd.concat(df_final, axis=1)
        if return_type=="dataframe":
            return df_final, encoders
    
        elif return_type=="tensors":
            features = df_final.drop(columns=["target", f"protected_{self.protected_col}"]).values
            targets = df_final["target"].values
            protected = df_final[f"protected_{self.protected_col}"].values
            return [torch.tensor(features).type(torch.float), torch.tensor(targets).type(torch.int), torch.tensor(protected).type(torch.int)], encoders


    def save_split_processed(self, split_ratio=(0.7,0.3), split_seed=42, return_type="tensors") :
        df = self.load_partial_processed()
        # get categorical encoders from full dataset
        encoders = self.get_categorical_encoders(df)
        # split
        train_frac, test_frac =split_ratio
        train_df, test_df = train_test_split(df, test_size=test_frac, random_state=split_seed, shuffle=True)
        # process train first to get numerical encoders
        train_data, encoders  = self.process_dataset(train_df, True, encoders, return_type)
        train_dict = {"x" : train_data[0], "y" : train_data[1].reshape(-1,1), "g" : train_data[2].reshape(-1,1)}
        test_data, _  = self.process_dataset(test_df, False, encoders,return_type)
        test_dict = {"x" : test_data[0], "y" : test_data[1].reshape(-1,1), "g" : test_data[2].reshape(-1,1)}
        dataset_dict = {"encoders" : encoders, "split_ratio" : split_ratio, "split_seed" : split_seed, "train" : train_dict, "test" : test_dict, }
        with open(f"{self.data_dir}/processed/procdata_{split_seed}.p","wb") as fi:
            pickle.dump(dataset_dict, fi)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()
    compas=CompasA()
    compas.save_split_processed(split_seed=args.seed)

