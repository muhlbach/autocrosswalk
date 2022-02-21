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
def load_example_data(which_data="default"):
    
    # PATHS
    THIS_DIR = __file__.rsplit("/", maxsplit=1)[0]+"/"
    DATA_DIR = os.path.join(THIS_DIR, "data/")
        
    if which_data=="default":
        df = pd.read_parquet(path=os.path.join(DATA_DIR,"example_data.parquet"))
    elif which_data=="missing":
        df = pd.read_parquet(path=os.path.join(DATA_DIR,"missing_data.parquet"))
    else:
        bg.exceptions.WrongInputException(input_name="which_data",
                                          provided_input=which_data,
                                          allowed_inputs=["default", "missing"])
    
    return df
