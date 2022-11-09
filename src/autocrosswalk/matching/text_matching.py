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
                 use_lower_characters=True,
                 ):
        self.method = method
        self.speed = speed
        self.use_lower_characters = use_lower_characters
        self.memory = {}
        
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
        bg.sanity_check.check_type(x=a,allowed=str,name="a")
        bg.sanity_check.check_type(x=b,allowed=list,name="b")
        
        # Correction
        if self.use_lower_characters:
            a = a.lower()
            b = [x.lower() for x in b]
            
        # Add to key if not present
        if not a in self.memory:
            self.memory[a] = {}
            
        # Find subset of b that needs to be estimated            
        b_not_estimated = [x for x in b if x not in self.memory.get(a)]
        
        # Estimate similarities
        if self.method=="Ratcliff-Obershelp":
            similarities_estimated = {x: self._ratcliff_obershelp(a=a,
                                                                  b=x,
                                                                  speed=self.speed,
                                                                  **kwargs) for x in b_not_estimated}
        
        # Increase memory
        self.memory[a] = {**self.memory[a],
                          **similarities_estimated}
            
        # Find similarities to be returned
        similarities_returned = [self.memory[a][x] for x in b]
            
        # if self.method=="Ratcliff-Obershelp":
        #     similarities = [self._ratcliff_obershelp(a=a,
        #                                              b=x,
        #                                              speed=self.speed,
        #                                              **kwargs) for x in b]
            
        return similarities_returned
            
            
        # This is way too slow!!!        
        # from diff_match_patch import diff_match_patch
        # P_temp[P_temp.index.get_level_values(key) == code_from] = [diff.diff_levenshtein(diffs=diff.diff_main(text1=code_from, text2=k)) / max(len(code_from),len(k)) for k in key_to]
        