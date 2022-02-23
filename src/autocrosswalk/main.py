#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import bodyguard as bg
import re

from .matching.text_matching import TextMatcher
from .matching.numeric_matching import NumericMatcher
from .matching.context_matching import ContextMatcher

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class AutoCrosswalk(object):
    """
    This class crosswalks between two documents 
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 n_best_match=2,
                 prioritize_exact_match=True,
                 enforce_completeness=True,
                 verbose=False):
        self.n_best_match = n_best_match
        self.prioritize_exact_match=prioritize_exact_match
        self.enforce_completeness=enforce_completeness
        self.verbose = verbose
        
        # Pre-set variables
        self.has_all_keys = False
        self.unique_from = []
        self.unique_to = []
        
        # Instantiate matchers
        self.numericmatcher = NumericMatcher()
        self.textmatcher = TextMatcher()
        self.contextmatcher = ContextMatcher(normalize=True,
                                             verbose=self.verbose,
                                             model_name_or_path="all-roberta-large-v1")

    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    TOL = 0.001

    # -------------------------------------------------------------------------
    # Very private functions
    # -------------------------------------------------------------------------            
    def __check_keys(self, keys, which_keys="text_key"):
        """
        Sanity check keys. We do not allow str, because we would have to overwrite the keys.
        """
        if keys is None:
            pass
        elif not isinstance(keys, list):
            raise Exception(f"When {which_keys} is not None, it must be an instance of list. Currently, it is {type(keys)}")
    

    def __fix_values(self,v):
        if isinstance(v, list):
            pass
        elif isinstance(v, str):
            v = [v]
        else:
            raise bg.exceptions.WrongInputTypeException(input_name="v",
                                                        provided_input=v,
                                                        allowed_inputs=[list, str])
        
        return v
    
    
    def __fix_by(self,by,cols):        
        if not isinstance(cols,list):
            raise bg.exceptions.WrongInputTypeException(input_name="cols",
                                                        provided_input=cols,
                                                        allowed_inputs=list)
        
        if by is None:
            by = cols
        elif isinstance(by, list):
            
            # Exclude elem from cols
            by = [x for x in by if not bg.tools.isin(a=x, b=cols)]
            
            by += cols
        elif isinstance(by, str):
            
            if bg.tools.isin(a=by, b=cols):
                by = cols
            else:            
                by = cols+[by]
        else:
            raise bg.exceptions.WrongInputTypeException(input_name="by",
                                                        provided_input=by,
                                                        allowed_inputs=[None, list, str])
        return by
    
    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
    def _check_keys(self, numeric_key, text_key, context_key):
        """
        Sanity check all keys
        """
        self.__check_keys(keys=numeric_key, which_keys="numeric_key")
        self.__check_keys(keys=text_key, which_keys="text_key")
        self.__check_keys(keys=context_key, which_keys="context_key")
        
        # Check if all are None
        if (numeric_key is None) and (text_key is None) and (context_key is None):
            raise Exception("Both 'numeric_key', 'text_key', 'context_key' cannot be None simultaneously")
            
        # If code passes until here, at least one key is different from None.
        if numeric_key is None:
            numeric_key = []
        if text_key is None:
            text_key = []
        if context_key is None:
            context_key = []            
            
        return numeric_key,text_key,context_key
            
    def _check_cols(self, df, cols):
        """
        Check if required columns are present
        """
        if not bg.tools.isin(a=cols, b=df.columns, how="all", return_element_wise=False):
            raise Exception(f"Some cols are not found in data. \nPresent columns: {df.columns.tolist()}. \nNecessary columns {cols}")

    def _check_prefix(self, prefix, df):
        """
        Check prefix
        """
        if not isinstance(prefix, str):
            raise bg.exceptions.WrongInputTypeException(input_name="prefix",
                                                        provided_input=prefix,
                                                        allowed_inputs=str)

    def _reindex_df(self, df, index=None, columns=None):
        """
        Reindex df by index and/or columns
        """
        if index is not None:        
            df = df.reindex(index=index)

        if columns is not None:        
            df = df.reindex(columns=columns)
        
        return df


    def _prepare_transition_matrix(self,
                                   df_from,
                                   df_to,
                                   numeric_key,
                                   text_key,
                                   context_key,
                                   fill_value=0):
        """
        Prepare an empty transition matrix
        """
        # Generate unique
        all_keys = numeric_key+text_key+context_key
        df_unique_from = df_from[all_keys].drop_duplicates()
        df_unique_to = df_to[all_keys].drop_duplicates()
        
        # Empty transition matrix
        transition_matrix = pd.DataFrame(data=fill_value,
                                         index=pd.MultiIndex.from_frame(df=df_unique_from),
                                         columns=pd.MultiIndex.from_frame(df=df_unique_to))
        
        return transition_matrix
        
    def _prepare_crosswalk(self,
                           df,
                           numeric_key,
                           text_key):
        """
        Prepare an empty crosswalk dataframe
        """
        # Generate unique
        df_unique = df[numeric_key+text_key].drop_duplicates()
        
        # Empty transition matrix
        crosswalk = pd.DataFrame(index=pd.MultiIndex.from_frame(df=df_unique,names=[f"FROM {c}" for c in df_unique.columns]),
                                 columns=[f"TO {c}" for c in df_unique.columns])

        crosswalk = pd.DataFrame(index=pd.MultiIndex.from_frame(df=df_unique),
                                 columns=df_unique.columns)
        
        return crosswalk
    

    
    
    def _compute_numeric_transition_prob(self,keys,transition_matrix):
        """
        Estimate transition probability matrix from numeric key
        """
        # Copy transition matrix
        P = transition_matrix.copy()
        P_temp = transition_matrix.copy()
        
        if keys:
            for key in keys:
                
                # Get unique keys to be matched                
                key_from = P.index.get_level_values(level=key).tolist()
                key_to = P.columns.get_level_values(level=key).tolist()
            
                for k in key_from:
                    P_temp[P_temp.index.get_level_values(key) == k] = self.numericmatcher.compute_similarity(a=k, b=key_to)
                 
                # Add to transition matrix
                P += P_temp
                
            # Normalize total to maximum 1
            P /= len(keys)
        
        return P
        
    def _compute_text_transition_prob(self,keys,transition_matrix):
        """
        Estimate transition probability matrix from text key
        """
        # Copy transition matrix
        P = transition_matrix.copy()
        P_temp = transition_matrix.copy()
        
        if keys:   
            for key in keys:
                
                # Get unique keys to be matched                
                key_from = P.index.get_level_values(level=key).tolist()
                key_to = P.columns.get_level_values(level=key).tolist()
            
                for k in key_from:
                    P_temp[P_temp.index.get_level_values(key) == k] = self.textmatcher.compute_similarity(a=k, b=key_to)
                 
                # Add to transition matrix
                P += P_temp
                
            # Normalize total to maximum 1
            P /= len(keys)
        
        return P
    
    def _compute_context_transition_prob(self,keys,transition_matrix):
        """
        Estimate transition probability matrix from context key
        """
        # Copy transition matrix
        P = transition_matrix.copy()
        P_temp = transition_matrix.copy()
                
        if keys:
            
            # Embed all keys
            new_keys = []

            for key in keys:
                
                # Get unique keys to be matched                
                key_from = P.index.get_level_values(level=key).tolist()
                key_to = P.columns.get_level_values(level=key).tolist()

                # Combine
                key_from_to = list(set(key_from+key_to))

                # Append
                new_keys += key_from_to
                
            # Next, make sure all of these context keys are embedded
            self.contextmatcher.embed_documents(documents=new_keys,
                                                return_embeddings=False)
            
            # Since all keys are now embedded, we com compute the similarity
            for key in keys:
                
                # Get unique keys to be matched                
                key_from = P.index.get_level_values(level=key).tolist()
                key_to = P.columns.get_level_values(level=key).tolist()
            
                for k in key_from:
                    P_temp[P_temp.index.get_level_values(key) == k] = self.contextmatcher.compute_similarity(a=k, b=key_to)
                 
                # Add to transition matrix
                P += P_temp
                
            # Normalize total to maximum 1
            P /= len(keys)
        
        return P    
    


    def _estimate_transition_matrix(self,
                                    numeric_key,
                                    text_key,
                                    context_key,
                                    transition_matrix_null):
        
        
        """
        Helper function to estimate transition matrices
        """
        n_transition_matrix = 0
        # Count text_key twice because we use this for context embeddings as well
        for mat in [numeric_key,text_key,context_key]:
            if mat:
                n_transition_matrix += 1
                
        # ---------------------------------------------------------------------
        # NUMERIC transition probability matrix
        # ---------------------------------------------------------------------
        if self.verbose>2:
            bg.tools.print2("Estimating NUMERIC transition matrix")
        
        transition_matrix_numeric = self._compute_numeric_transition_prob(keys=numeric_key,
                                                                          transition_matrix=transition_matrix_null)

        # Sort on both index and column according to null matrix
        transition_matrix_numeric = self._reindex_df(df=transition_matrix_numeric,
                                                     index=transition_matrix_null.index,
                                                     columns=transition_matrix_null.columns)

        # Check if each FROM_KEY has an exact numeric match. If this is the case, no need to compute transition matrix for text keys
        has_exact_numeric_match = (transition_matrix_numeric==1).sum(axis=1)

        # ---------------------------------------------------------------------
        # TEXT transition probability matrix
        # ---------------------------------------------------------------------
        if has_exact_numeric_match.all():
            # All keys have exact match. Thus, copy another transition matrix and fill with 1
            transition_matrix_text = transition_matrix_null.copy()
            transition_matrix_text.iloc[:,:] = 1
        else:
            # Not all keys have exact matches. Find those who have and those who have not.
            mask_exact = has_exact_numeric_match>=1
            
            # Extract transition matrix for exact matches and set probability to 1
            transition_matrix_text_exact = transition_matrix_null.loc[mask_exact].copy()
            transition_matrix_text_exact.loc[:,:] = 1
            
            # Extract transition matrix for non-exact matches and compute probability
            transition_matrix_null_non_exact = transition_matrix_null.loc[~mask_exact]
            if self.verbose>2:
                bg.tools.print2("Estimating TEXT transition matrix")    
            transition_matrix_text_non_exact = self._compute_text_transition_prob(keys=text_key,
                                                                        transition_matrix=transition_matrix_null_non_exact)

            # Combine            
            transition_matrix_text = pd.concat(objs=[transition_matrix_text_exact,
                                                     transition_matrix_text_non_exact],
                                               axis=0)
        
        # Sort on both index and column according to null matrix
        transition_matrix_text = self._reindex_df(df=transition_matrix_text,
                                                  index=transition_matrix_null.index,
                                                  columns=transition_matrix_null.columns)
            
        # ---------------------------------------------------------------------
        # CONTEXT transition probability matrix
        # ---------------------------------------------------------------------
        # Check if each key has an exact match (either numeric or text). We only compute context similarity if no exact match is available
        has_exact_numeric_text_match = (transition_matrix_numeric==1).sum(axis=1) + (transition_matrix_text==1).sum(axis=1)
        
        if has_exact_numeric_text_match.all():
            # All keys have exact match. Thus, copy another transition matrix and fill with 1
            transition_matrix_context = transition_matrix_null.copy()
            transition_matrix_context.iloc[:,:] = 1
        else:
            # Not all keys have exact matches. Find those who have and those who have not.
            mask_exact = has_exact_numeric_text_match>=1
            
            # Extract transition matrix for exact matches and set probability to 1
            transition_matrix_context_exact = transition_matrix_null.loc[mask_exact].copy()
            transition_matrix_context_exact.loc[:,:] = 1
            
            # Extract transition matrix for non-exact matches and compute probability
            transition_matrix_null_non_exact = transition_matrix_null.loc[~mask_exact]
            
            if self.verbose>2:
                bg.tools.print2("Estimating CONTEXT transition matrix")    
            transition_matrix_context_non_exact = self._compute_context_transition_prob(keys=context_key,
                                                                                        transition_matrix=transition_matrix_null_non_exact)

            # Combine            
            transition_matrix_context = pd.concat(objs=[transition_matrix_context_exact,
                                                        transition_matrix_context_non_exact],
                                               axis=0)
        
        # Sort on both index and column according to null matrix
        transition_matrix_context = self._reindex_df(df=transition_matrix_context,
                                                     index=transition_matrix_null.index,
                                                     columns=transition_matrix_null.columns)
        
        # At this point, we are DONE estimating transition matrices. Let's sanity check and combine!
        
        # Sanity check indices
        if (transition_matrix_numeric.index != transition_matrix_text.index).any():
            raise Exception("Numeric and text transition matrices have different indices")
        if (transition_matrix_numeric.index != transition_matrix_context.index).any():
            raise Exception("Numeric and context transition matrices have different indices")
            
        # Combine transition matrices
        transition_matrix_average = transition_matrix_numeric + transition_matrix_text + transition_matrix_context
        
        # Normalize
        transition_matrix_average /= n_transition_matrix
        
        transition_matrix = {
            "numeric" : transition_matrix_numeric,
             "text" : transition_matrix_text,
             "context" : transition_matrix_context,
             "average" : transition_matrix_average
             }
        
        return transition_matrix

    def _find_exact_match(self,transition_matrix,prioritize_numeric_match=True):
        
        """
        Find exact matches
        """            
        if not isinstance(transition_matrix,dict):
            raise bg.exceptions.WrongInputTypeException(input_name="transition_matrix",
                                                        provided_input=transition_matrix,
                                                        allowed_inputs=dict)

        # Find exact matches by numeric and text keys
        mask_exact_numeric = transition_matrix["numeric"] >= 1-self.TOL
        mask_exact_text = transition_matrix["text"] >= 1-self.TOL
        
        # Combine numeric and text
        mask_exact = mask_exact_numeric | mask_exact_text
        
        # If there are more than 1 exact match, it is typically because two codes have the same text label.
        mask_double = mask_exact.sum(axis=1)>1
        
        # Thus, we overwrite the exact matches, but allow the user to choose between numeric or text matches
        if mask_double.sum()>0:        
            if prioritize_numeric_match:        
                mask_exact.loc[mask_double] = mask_exact_numeric.loc[mask_double]
            else:
                mask_exact.loc[mask_double] = mask_exact_text.loc[mask_double]
        
        # Construct temporary matrix of exact matches (a series of dataframes)
        temp = mask_exact.apply(lambda x: x.index[x == True].to_frame(index=False), axis=1)
        
        # Find empty dataframes (those that do not match exactly)
        mask_empty = pd.Series(data=[temp.loc[idx].empty for idx in temp.index],
                               index=temp.index)
        
        # Subset temporaty
        temp = temp.loc[~mask_empty]
        
        # Turn series of dataframes into dataframe by concatenating
        exact = pd.concat(objs=temp.tolist())

        # Fix index
        exact.index = temp.index

        return exact

    def _find_nbest_match(self,transition_matrix,n=5):
    
        if not isinstance(transition_matrix,pd.DataFrame):
            raise bg.exceptions.WrongInputTypeException(input_name="transition_matrix",
                                                        provided_input=transition_matrix,
                                                        allowed_inputs=pd.DataFrame)
    
    
        # Find n closest numeric matches
        temp = transition_matrix.apply(lambda s, n: s.nlargest(n).index.to_frame(index=False),
                                       axis=1,
                                       n=n)

        nbest = pd.concat(objs=temp.tolist())

        # Fix index by repeating 
        nbest.index = temp.index.repeat(repeats=n)
        
        return nbest
    
    
    def _transpose_transition_matrix(self,transition_matrix):
        
        transition_matrix_transposed = {k:v.T for k,v in transition_matrix.items()}
        
        return transition_matrix_transposed
         
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def estimate_transition_matrix(self,
                                   df_from,
                                   df_to,
                                   use_existing_transition_matrix=True,
                                   numeric_key=None,
                                   text_key=None,
                                   context_key=None):
        """
        Estimate transition probability
        Note that if 'use_existing_transition_matrix=True', it means that we OVERWRITE existing transition matrices if they do not have ALL unique keys
        """        
        # ---------------------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------------------
        numeric_key,text_key,context_key = self._check_keys(numeric_key=numeric_key,           
                                                            text_key=text_key,
                                                            context_key=context_key)
        
        all_keys = numeric_key+text_key+context_key
        self._check_cols(df=df_from,cols=all_keys)
        self._check_cols(df=df_to,cols=all_keys)
        # ---------------------------------------------------------------------
        # Prepare
        # ---------------------------------------------------------------------
        # Prepare a standard transition matrix
        transition_matrix_null = self._prepare_transition_matrix(df_from=df_from,
                                                                 df_to=df_to,
                                                                 numeric_key=numeric_key,
                                                                 text_key=text_key,
                                                                 context_key=context_key,
                                                                 fill_value=0)
        
        if use_existing_transition_matrix:                        
            # If we are using an existing transition matrix, we need to make sure that all unique keys are present
            # If not all keys are present, we overwrite exisiting transition matrix
            
            # Get current unique from and to
            unique_from = transition_matrix_null.index
            unique_to = transition_matrix_null.columns

            # Check if they are present in existing
            has_from_keys = all(k in self.unique_from for k in unique_from)
            has_to_keys = all(k in self.unique_to for k in unique_to)                
        
            if (has_from_keys and has_to_keys):
                self.has_all_keys = True
            else:
                self.has_all_keys = False
                

            if not self.has_all_keys:
                
                if self.verbose:
                    bg.exceptions.print_warning("Some keys are missing in existing transition matrix. \nRe-estimating transition matrix and overwriting existing")
                
                # Some keys are missing. Hence, we estimate transition matrices again
                self.transition_matrix = self._estimate_transition_matrix(numeric_key=numeric_key,
                                                                          text_key=text_key,
                                                                          context_key=context_key,
                                                                          transition_matrix_null=transition_matrix_null)

                # Overwrite keys
                self.unique_from = unique_from
                self.unique_to = unique_to

            # Get transition matrix          
            transition_matrix = self.transition_matrix
                
        else:
            # Estimate new probability matrix
            transition_matrix = self._estimate_transition_matrix(numeric_key=numeric_key,
                                                                 text_key=text_key,
                                                                 context_key=context_key,
                                                                 transition_matrix_null=transition_matrix_null)
            


                
        return transition_matrix
                
    
    
    def generate_crosswalk(self,
                           df_from,
                           df_to,
                           numeric_key=None,
                           text_key=None,
                           context_key=None,
                           prefix="NEW "):
        """
        Prepare proper crosswalk file
        """
        # ---------------------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------------------
        numeric_key,text_key,context_key = self._check_keys(numeric_key=numeric_key,           
                                                            text_key=text_key,
                                                            context_key=context_key)
        
        all_keys = numeric_key+text_key+context_key
        self._check_cols(df=df_from,cols=all_keys)
        self._check_cols(df=df_to,cols=all_keys)
        
        self._check_prefix(prefix=prefix, df=df_from)
        # ---------------------------------------------------------------------
        # Estimate transition proabability matrix
        # ---------------------------------------------------------------------
        transition_matrix = self.estimate_transition_matrix(df_from=df_from,
                                                            df_to=df_to,
                                                            use_existing_transition_matrix=True,
                                                            numeric_key=numeric_key,
                                                            text_key=text_key,
                                                            context_key=context_key)
                
        # ---------------------------------------------------------------------
        # Make crosswalk
        # ---------------------------------------------------------------------
        # Pre-allocate list of crosswalks to be 
        crosswalks = []
        
        crosswalk = self._prepare_crosswalk(df=df_from,
                                            numeric_key=numeric_key,
                                            text_key=text_key)
        
        if self.prioritize_exact_match:
            # Find exact matches
            crosswalk_exact = self._find_exact_match(transition_matrix=transition_matrix)
            
            # Update existing crosswalk
            crosswalk.update(other=crosswalk_exact)
            
            crosswalks.append(crosswalk_exact)
            
        ## Find nbest matches
        # Mask those not found by now
        mask_find = crosswalk.isna().any(axis=1)
        mask_lookup = transition_matrix["average"].index.isin(crosswalk.loc[mask_find].index)

        if mask_lookup.sum()>0:
            # Find nbest candidates
            crosswalk_nbest = self._find_nbest_match(transition_matrix=transition_matrix["average"].loc[mask_lookup],
                                                     n=self.n_best_match)
    
            crosswalks.append(crosswalk_nbest)


        if self.enforce_completeness:
            
            # Transpose all transition matrices
            transition_matrix_to = self._transpose_transition_matrix(transition_matrix=transition_matrix)
            
            # Make sure that all keys in "df_to" are also matched
            crosswalk_to = self._prepare_crosswalk(df=df_to,
                                                   numeric_key=numeric_key,
                                                   text_key=text_key)
                
            # Find exact matches
            crosswalk_exact_to = self._find_exact_match(transition_matrix=transition_matrix_to)
            
            # Update existing crosswalk
            crosswalk_to.update(other=crosswalk_exact_to)
            
            mask_find = crosswalk_to.isna().any(axis=1)
            mask_lookup = transition_matrix_to["average"].index.isin(crosswalk_to.loc[mask_find].index)

            if mask_lookup.sum()>0:

                # Find nbest candidates
                crosswalk_nbest_to_temp = self._find_nbest_match(transition_matrix=transition_matrix_to["average"].loc[mask_lookup],
                                                         n=self.n_best_match)
    
                # Flip index and columns (not transport, but just swap)
                crosswalk_nbest_to = crosswalk_nbest_to_temp.index.to_frame(index=False) 
                crosswalk_nbest_to.index = pd.MultiIndex.from_frame(df=crosswalk_nbest_to_temp)
    
                crosswalks.append(crosswalk_nbest_to)
            
        # Collect all crosswalks
        df_crosswalk = pd.concat(objs=crosswalks) 

        # Update names
        df_crosswalk.columns = [prefix+c for c in df_crosswalk.columns]
        
        # Reset index
        df_crosswalk.reset_index(inplace=True)
        
        # Drop duplicates
        df_crosswalk.drop_duplicates(inplace=True)
        
        # Reindex
        df_crosswalk.set_index(keys=numeric_key+text_key, inplace=True)
        

        return df_crosswalk
        
    
        
    def perform_crosswalk(self,
                          crosswalk,
                          df,
                          values,
                          prefix="NEW ",
                          by=None):
        """
        Perform crosswalk
        """
        self._check_prefix(prefix=prefix, df=df)
        
        # Sanity check
        values = self.__fix_values(v=values)
        self._check_cols(df=df, cols=values)
        
        # Find columns to merge by
        merge_cols = list(crosswalk.index.names)
        merge_cols.sort()
        
        # Fix by
        self.by = self.__fix_by(by=by,cols=merge_cols)
        
        # Initial merge 
        df_merged = df.merge(right=crosswalk,
                             how='left',
                             on=merge_cols,
                             left_index=False,
                             right_index=False,
                             sort=False,
                             suffixes=('_x', '_y'),
                             copy=True,
                             indicator=False,
                             validate=None)

        # Drop merged cols
        df_merged.drop(columns=merge_cols,
                       inplace=True)

        # Rename
        df_merged.columns = [re.sub(pattern=prefix, repl="", string=c) for c in df_merged.columns]

        # Aggregate      
        df_agg = df_merged.groupby(by=self.by,as_index=False,sort=True)[values].mean()

        return df_agg
        
    def impute_missing(self,
                       df,
                       numeric_key,
                       text_key,
                       values,
                       by=None):
        """
        Impute missing values by averaging closest matches 
        """
        # Break link
        df = df.copy()
        
        # Sanity check
        values = self.__fix_values(v=values)
        self._check_cols(df=df, cols=values)

        # Mask missing
        mask_na = df[values].isna().any(axis=1)
        
        # Only impute in case of missingness
        if mask_na.sum()>0:
            
            # Subset
            df_from = df.loc[~mask_na]
            
            # Generate crosswalk
            crosswalk = self.generate_crosswalk(df_from=df_from,
                                                df_to=df,
                                                numeric_key=numeric_key,
                                                text_key=text_key)
            
            # Perform crosswalk
            df_updated = self.perform_crosswalk(crosswalk=crosswalk,
                                                df=df_from,
                                                values=values,
                                                by=by)
        
        
            # Set index
            df.set_index(keys=self.by, inplace=True)
            df_updated.set_index(keys=self.by, inplace=True)
            
            # Update
            df.update(other=df_updated, overwrite=True)
            
            # Reset index
            df.reset_index(drop=False, inplace=True)
        
        return df
    