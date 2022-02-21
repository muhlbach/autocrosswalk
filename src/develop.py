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
from autocrosswalk.tools import load_example_data
#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# DEFAULT DATA
#------------------------------------------------------------------------------
# Load example data
data = load_example_data(which_data="default")
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
                                                text_key=['Job title'])

# Perform crosswalk
df_updated = autocrosswalk.perform_crosswalk(crosswalk=df_crosswalk,
                                             df=data_from,
                                             values=["Data Value"],
                                             by=['Date', 'DB',
                                                 'Category', 'Element ID', 'Element Name','Element description'])


#------------------------------------------------------------------------------
# MISSING DATA
#------------------------------------------------------------------------------
# Load example data
data = load_example_data(which_data="missing")
wage_cols = ["h_mean", "a_mean"]
data_from = data.loc[~data[wage_cols].isna().any(axis=1)]     
data_to = data.copy()

# Instantiate
self = autocrosswalk = AutoCrosswalk(n_best_match=3,
                              prioritize_exact_match=True,
                              enforce_completeness=True,
                              verbose=3)

# Generate crosswalk file
df_crosswalk = autocrosswalk.generate_crosswalk(df_from=data_from,
                                                df_to=data_to,
                                                numeric_key=['occ_code'],
                                                text_key=['occ_title'])

# Perform crosswalk
df_updated = autocrosswalk.perform_crosswalk(crosswalk=df_crosswalk,
                                             df=data_from,
                                             values=wage_cols)

# Show how to impute missing
df = autocrosswalk.impute_missing(df=data,
                                  numeric_key=['occ_code'],
                                  text_key=['occ_title'],
                                  values=wage_cols)









