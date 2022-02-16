#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import os
import pandas as pd

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def load_example_data():
    
    # PATHS
    THIS_DIR = __file__.rsplit("/", maxsplit=1)[0]+"/"
    DATA_DIR = os.path.join(THIS_DIR, "data/")
        
    df = pd.read_parquet(path=os.path.join(DATA_DIR,"example_data.parquet"))
    
    return df
