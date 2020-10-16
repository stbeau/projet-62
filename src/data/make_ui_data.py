# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:17:25 2020

@author: sbeau
"""

import os
import string
import gzip
import shutil
#import janitor
import pandas as pd
import pickle


INTERIM_PATH = "../../data/interim"
FILENAME = "dvf_2019.csv"

INPUT_FULLNAME = os.path.join(INTERIM_PATH,FILENAME)
OUTPUT_FULLNAME = "../../data/ui/ui_data.pkl"



def export_features_data(input_fullname, output_filename):
    
    all_data = pd.read_csv(input_fullname, low_memory=False)
    all_data["code_postal"] = all_data["code_postal"].astype("category").astype("str")
    
    code_postal_ = list(all_data["code_postal"].unique())
    code_commune_ = list(all_data["code_commune"].unique())
    
    ui_data = {"code_postal": code_postal_, "code_commune":code_commune_}
    
    with open(output_filename, "wb") as handle:
        pickle.dump(ui_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
    
    return



if __name__ == '__main__':
    
    export_features_data(INPUT_FULLNAME, OUTPUT_FULLNAME)    