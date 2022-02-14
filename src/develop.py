"""
This script implements some basic tests of the package that should be run before uploading
"""
#------------------------------------------------------------------------------
# Run interactively
#------------------------------------------------------------------------------
# import os
# # Manually set path of current file
# path_to_here = "/Users/muhlbach/Repositories/autocrosswalk/src/"
# # Change path
# os.chdir(path_to_here)
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import bodyguard as bg

from autocrosswalk.main import AutoCrosswalk

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
# Load example data
data = pd.read_parquet(path="autocrosswalk/data/data.parquet")

data_from = data.loc[data["DB"]=="db_20_0"]
data_to = data.loc[data["DB"]=="db_26_1"]


# Instantiate
autocrosswalk = AutoCrosswalk()


autocrosswalk.perform_crosswalk(df_from=data_from,
                                df_to=data_to,
                                values="Data Value")
