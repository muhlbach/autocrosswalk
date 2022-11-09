#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import bodyguard as bg
from econnlp.embedding.docs import DocumentEmbedder

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class ContextMatcher(DocumentEmbedder):
    """
    Match context
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)
        self.memory = {}
        
    # -------------------------------------------------------------------------
    # Public function
    # -------------------------------------------------------------------------        
    def compute_similarity(self, a, b, **kwargs):
        
        # Sanity checks
        bg.sanity_check.check_type(x=a,allowed=list,name="a")
        bg.sanity_check.check_type(x=b,allowed=list,name="b")
                
        # Add to keys if not present
        for y in a:
            if not y in self.memory:
                self.memory[y] = {}
                
        # Find subset of a that needs to be estimated
        a_not_estimated = [y for y in a if not all(x in self.memory[y] for x in b)]

        # Find subset of b that needs to be estimated
        b_not_estimated = [x for x in b if not all(x in self.memory[y] for y in a)]

        # Compute similarities
        similarities_estimated = super().compute_similarity(a=a_not_estimated,
                                                            b=b_not_estimated,
                                                            **kwargs)

        # Increase memory        
        self.memory = {**self.memory,
                       **similarities_estimated.to_dict()}
        
        # Convert back to dataframe
        similarities_all = pd.DataFrame().from_dict(self.memory)

        # Subset
        # Find similarities to be returned
        similarities_returned = similarities_all.loc[a,b]

        return similarities_returned
