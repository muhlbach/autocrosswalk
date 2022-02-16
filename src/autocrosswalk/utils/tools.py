#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import os
import pandas as pd
import bodyguard as bg


#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def load_example_data():
    
    print(__file__)
    
    # PATHS
    MAIN_DIR = __file__.rsplit("/", maxsplit=1)[0]+"/"
    DATA_DIR = os.path.join(MAIN_DIR, "data/")
    
    print(MAIN_DIR)
    print(DATA_DIR)
    
    
load_example_data()