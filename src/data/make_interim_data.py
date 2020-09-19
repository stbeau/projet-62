# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:17:25 2020

@author: sbeau
"""

import os
import string
import janitor
import pandas as pd


RAW_PATH = "../../data/raw"
INTERIM_PATH = "../../data/interim"
FILENAME = "valeursfoncieres-2019.txt"

INPUT_FULLNAME = os.path.join(RAW_PATH,FILENAME)
OUTPUT_FULLNAME = os.path.join(INTERIM_PATH,".".join([os.path.splitext(FILENAME)[0],"csv"]))

def duplicate_raw_data_to_default_csv(input_fullname=INPUT_FULLNAME, output_fullname=OUTPUT_FULLNAME):
    
    raw_data = pd.read_csv(input_fullname, delimiter="|", encoding='utf-8', infer_datetime_format=True, decimal=",", low_memory=False)
    raw_data.clean_names(remove_special=True).to_csv(output_fullname, index=False)
    return


def split_data_on_type_local(input_fullname, output_filepath):
    
    input_filename = os.path.splitext(os.path.basename(input_fullname))[0]
    
    raw_data = pd.read_csv(input_fullname, low_memory=False)
    types_locaux = raw_data["type_local"].unique()
    
    for type_local in types_locaux:
        output_filename = "".join([input_filename,"-",format_filename(type_local),".csv"])
        output_fullname = os.path.join(output_filepath, output_filename)
        #raw_data[raw_data["type_local"]]==type_local].drop(columns="type_local").to_csv(output_fullname)
        print(type_local)
    return

    #os.path.dirname
    #raw_data.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')


def make_interim_data():
    #raw_data.clean_names(remove_special=True).
    return

def format_filename(s):
    # valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    # Retirer les parenthèse ( et ) des caractères permis dans un nom de fichier
    valid_chars = "-_ %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_')
    return filename


if __name__ == '__main__':
    
   # duplicate_raw_data_to_default_csv(input_fullname=INPUT_FULLNAME, output_fullname=OUTPUT_FULLNAME)
   split_data_on_type_local(input_fullname=OUTPUT_FULLNAME, output_filepath=INTERIM_PATH)
    