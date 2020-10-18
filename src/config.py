from pathlib import Path
import os


class Config:
    
    SRC_PATH = "../"
    
    DOWNLOAD_PATH = "https://cadastre.data.gouv.fr/data/etalab-dvf/latest/csv/2019/"
    INPUT_FILENAME = "full.csv.gz"
    OUTPUT_PATH = "../data/raw"
    OUTPUT_FILENAME = "dvf_2019.gz"

    SOURCE_URL = os.path.join(DOWNLOAD_PATH,INPUT_FILENAME)
    OUTPUT_FULLNAME = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)


    METRIC_FILE_PATH = SRC_PATH / "metrics.json"



