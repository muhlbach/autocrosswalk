#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def compute_similarity(self,a,b,**kwargs):
        """
        Compute similarity between contexts
        """
        similarities = self.compute_similarity_between_two_embeddings(a=a,
                                                                      b=b,
                                                                      metric="cosine",
                                                                      **kwargs)
            
        return similarities
   