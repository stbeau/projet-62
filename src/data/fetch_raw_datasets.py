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

URL_STABLE = "https://www.data.gouv.fr/fr/datasets/r/1be77ca5-dc1b-4e50-af2b-0240147e0346"
DOWNLOAD_ROOT = "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20200416-115822/valeursfoncieres-2019.txt"
FILENAME = "valeursfoncieres-2019.txt" #os.path.join("datasets", "housing")

SOURCE_URL = os.path.join(DOWNLOAD_ROOT,FILENAME)
OUTPUT_PATH = "../../data/raw"
OUTPUT_FILENAME = os.path.join(OUTPUT_PATH,FILENAME)


def fetch_raw_datasets(input_filelink=SOURCE_URL, output_filepath=OUTPUT_PATH, filename=FILENAME):

    output_filename = os.path.join(output_filepath,filename)

    if not os.path.isdir(output_filepath):
        os.makedirs(output_filepath)
        print(f"Création du répertoire : {output_filepath}")
    
    file_stream = requests.get(input_filelink, stream=True, allow_redirects=True)
    
    try :
        print(f"Input filelink : {input_filelink}")
        print(f"Output filename : {output_filename}")
        with open(output_filename, 'wb') as local_file:
            for data in file_stream:
                local_file.write(data)
            print("Terminé !")
    except FileExistsError as e:
        print(f'Le fichier existe déjà ! : {e}')

    return



if __name__ == '__main__':
    
    url_path = ""
    file_name = ""
    fetch_raw_datasets(input_filelink=URL_STABLE, output_filepath=OUTPUT_PATH, filename=FILENAME)



