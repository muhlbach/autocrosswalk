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
autocrosswalk = AutoCrosswalk(n_best_match=3,
                              prioritize_exact_match=True,
                              enforce_completeness=True,
                              verbose=2)

# Generate crosswalk file
df_crosswalk = autocrosswalk.generate_crosswalk(df_from=data_from,
                                                df_to=data_to,
                                                numeric_key=['O*NET-SOC Code'],
                                                numeric_type="categorical",
                                                text_key=['Job title'])

# Perform crosswalk
df_updated = autocrosswalk.perform_crosswalk(crosswalk=df_crosswalk,
                                             df=data_from,
                                             values=["Data Value"],
                                             by=['Date', 'DB',
                                                 'Category', 'Element ID', 'Element Name','Element description', 'Job description'])

len(df_updated["Job title"].unique())
