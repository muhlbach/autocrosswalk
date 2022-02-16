
# *** ATTENTION ***
Don't immidiately run `pip install autocrosswalk`. See Section _Installation_.

# autocrosswalk: A generic approach to crosswalking

Please contact the authors below if you find any bugs or have any suggestions for improvement. Thank you!

Author: Nicolaj Søndergaard Mühlbach (n.muhlbach at gmail dot com, muhlbach at mit dot edu) 

## Code dependencies
This code has the following dependencies:

- Python >=3.6
- numpy >=1.19
- pandas >=1.3

## Installation

## Usage

```python
# Libraries
import pandas as pd
from autocrosswalk.main import AutoCrosswalk

# Instantiate
autocrosswalk = AutoCrosswalk(n_best_match=3,
                              prioritize_exact_match=True,
                              enforce_completeness=True,
                              verbose=2)

# Load example data
data = pd.read_parquet(path="autocrosswalk/data/data.parquet")

# Separate into old and new data, i.e., we crosswalk the 'data_from' to 'data_to' 
data_from = data.loc[data["DB"]=="db_20_0"]
data_to = data.loc[data["DB"]=="db_26_1"]

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

# Now, 'df_updated' has all new keys from 'data_to'!
```
<!-- ## Example
We provide an example script in `demo.py`. -->
