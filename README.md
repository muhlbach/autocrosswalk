
# autocrosswalk: A generic approach to crosswalking

This library automates crosswalks from one dataframe to another.

Please contact the authors below if you find any bugs or have any suggestions for improvement. Thank you!

Author: Nicolaj Søndergaard Mühlbach (n.muhlbach at gmail dot com, muhlbach at mit dot edu) 

## Code dependencies
This code has the following dependencies:
- Python >=3.6
- pandas >=1.3

## Installation
There are no heavy dependencies for this library to work. We have included an example that requires a parquet reader, e.g., `pyarrow`, `brotli`, or `fastparquet`. One needs to have one of them installed in order to use the example data provided.
Otherwise, go ahead and install by `pip install autocrosswalk`.
## Usage

```python
# Libraries
from autocrosswalk.main import AutoCrosswalk
from autocrosswalk.tools import load_example_data

# Load example data
data = load_example_data()

# Separate into old and new data, i.e., we crosswalk the 'data_from' to 'data_to' 
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
                                                 'Category', 'Element ID',
                                                 'Element Name','Element description'])

# Check if number of unique keys match
print(len(df_updated["O*NET-SOC Code"].unique()) == len(data_to["O*NET-SOC Code"].unique()))
print(len(df_updated["Job title"].unique()) == len(data_to["Job title"].unique()))

# Now, 'df_updated' has all new keys from 'data_to'!
```
<!-- ## Example
We provide an example script in `demo.py`. -->
