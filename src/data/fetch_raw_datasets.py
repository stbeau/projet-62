# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:02:59 2020

@author: sbeau
"""
# Problèmes à corriger
# 1. Si le fichier existe déjà, la fonction écrit par-dessus, donc le try except ne fonctionne pas.
# 2. Le suivi d'une redirection ne fonctionne pas donc allow_redirects=True n'a pas l'effet escompté.
#

import os
import requests

def change_filename_in_filelink(filelink, new_filename):
    dirname = os.path.dirname(filelink)
    file_ext = os.path.splitext(os.path.basename(filelink))[1]
    return "".join([dirname,new_filename,file_ext])


DOWNLOAD_PATH = "https://cadastre.data.gouv.fr/data/etalab-dvf/latest/csv/2019/"
INPUT_FILENAME = "full.csv.gz"
OUTPUT_PATH = "../../data/raw"
OUTPUT_FILENAME = "dvf_2019.gz"

SOURCE_URL = os.path.join(DOWNLOAD_PATH,INPUT_FILENAME)
OUTPUT_FULLNAME = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)


def fetch_raw_datasets(input_filelink=SOURCE_URL, output_fullname=OUTPUT_FULLNAME):

    output_filepath = os.path.dirname(output_fullname)

    if not os.path.isdir(output_filepath):
        os.makedirs(output_filepath)
        print(f"Création du répertoire : {output_filepath}")
    
    file_stream = requests.get(input_filelink, stream=True, allow_redirects=True)
    
    try :
        print(f"Input filelink : {input_filelink}")
        print(f"Output fullname : {output_fullname}")
        with open(output_fullname, 'wb') as local_file:
            for data in file_stream:
                local_file.write(data)
            print("Terminé !")
    except FileExistsError as e:
        print(f'Le fichier existe déjà ! : {e}')

    return



if __name__ == '__main__':
    
    fetch_raw_datasets(input_filelink=SOURCE_URL, output_fullname=OUTPUT_FULLNAME)



