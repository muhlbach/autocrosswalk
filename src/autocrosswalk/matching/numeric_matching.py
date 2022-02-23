#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import bodyguard as bg

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
    def __init__(self):
        pass
        
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
    def _check_a(self, a):
        if not isinstance(a,str):
            raise bg.exceptions.WrongInputTypeException(input_name="a",
                                                        provided_input=a,
                                                        allowed_inputs=str)

    def _check_b(self, b):
        if not isinstance(b,list):
            raise bg.exceptions.WrongInputTypeException(input_name="b",
                                                        provided_input=b,
                                                        allowed_inputs=list)
        
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
        self._check_a(a=a)
        self._check_b(b=b)
        
        # Compute similarities for all elements in b
        similarities = [self._compare_codes(a=a,b=x) for x in b]
            
        return similarities
            
        