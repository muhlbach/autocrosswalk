#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import bodyguard as bg
import re
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class NumericMatcher(object):
    """
    Match numeric
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 remove_special_characters=True):
        self.remove_special_characters = remove_special_characters
        self.memory = {}
        
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
    def _compare_codes(self,a,b):
        # Adjust lenghs
        if len(a)==len(b):
            pass
        elif len(a)>len(b):
            b = b.ljust(len(a), 'X')
        elif len(a)<len(b):
            a = a.ljust(len(b), 'X')
            
        # Compute len
        len_str = len(a)
        
        # Initial score
        score = a==b

        for r in range(1,len_str):
            score += a[:-r] == b[:-r]
            
        # Normalize score
        score = score / len_str
        
        return score
        
        
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def compute_similarity(self,a,b):
        """
        Compute similarity between string and each element in list
        """
        # Sanity checks
        bg.sanity_check.check_type(x=a,allowed=str,name="a")
        bg.sanity_check.check_type(x=b,allowed=list,name="b")
        
        if self.remove_special_characters:
            a = re.sub('\W+','', a)
            b = [re.sub('\W+','', x) for x in b]
            
        # Add to key if not present
        if not a in self.memory:
            self.memory[a] = {}
            
        # Find subset of b that needs to be estimated            
        b_not_estimated = [x for x in b if x not in self.memory.get(a)]
        
        # Estimate similarities
        similarities_estimated = {x: self._compare_codes(a=a,b=x) for x in b_not_estimated}
        
        # Increase memory
        self.memory[a] = {**self.memory[a],
                          **similarities_estimated}
            
        # Find similarities to be returned
        similarities_returned = [self.memory[a][x] for x in b]
                        
        return similarities_returned    
            
        