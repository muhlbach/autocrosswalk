#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import bodyguard as bg
from difflib import SequenceMatcher
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class TextMatcher(object):
    """
    Match text
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 method="Ratcliff-Obershelp",
                 speed="slow",
                 ):
        self.method = method
        self.speed = speed
        
        if not bg.tools.isin(a=self.method, b=self.METHOD_OPT):
            bg.exceptions.WrongInputException(input_name="method",
                                              provided_input=self.method,
                                              allowed_inputs=self.METHOD_OPT)
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    METHOD_OPT = ["Ratcliff-Obershelp"]
    SPEED_OPT = ["slow", "quick", "real_quick"]
    
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

    
    def _ratcliff_obershelp(self,a,b,speed="slow",isjunk=None,autojunk=True):
        
        
        if speed=="slow":        
            seq_similarity = SequenceMatcher(isjunk=isjunk,
                                             a=a,
                                             b=b,
                                             autojunk=autojunk).ratio()
        elif speed=="quick":
            seq_similarity = SequenceMatcher(isjunk=isjunk,
                                             a=a,
                                             b=b,
                                             autojunk=autojunk).quick_ratio()
        elif speed=="real_quick":
            seq_similarity = SequenceMatcher(isjunk=isjunk,
                                             a=a,
                                             b=b,
                                             autojunk=autojunk).real_quick_ratio()
        else:
            raise bg.exceptions.WrongInputException(input_name="speed",
                                                    provided_input=speed,
                                                    allowed_inputs=self.SPEED_OPT)
            
        return seq_similarity
            
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def compute_similarity(self,a,b,**kwargs):
        """
        Compute similarity between string and each element in list
        """
        # Sanity checks
        self._check_a(a=a)
        self._check_b(b=b)
        
        if self.method=="Ratcliff-Obershelp":
            similarities = [self._ratcliff_obershelp(a=a,
                                                     b=x,
                                                     speed=self.speed,
                                                     **kwargs) for x in b]
            
            
        return similarities
            
            
        
# from diff_match_patch import diff_match_patch
# P_temp[P_temp.index.get_level_values(key) == code_from] = [diff.diff_levenshtein(diffs=diff.diff_main(text1=code_from, text2=k)) / max(len(code_from),len(k)) for k in key_to]
        