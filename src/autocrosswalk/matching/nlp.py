#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import os
import pandas as pd
import numpy as np
import bodyguard as bg

# Using SentenceTransformers NLP library
from sentence_transformers import SentenceTransformer

# Using HuggingFace NLP library
from transformers import AutoTokenizer, AutoModel
import torch

# From this library
from ..utils.convert import convert_dict_to_df, convert_df_to_dict
from ..utils.distance import normalize_by_norm
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class DocumentEmbedder(object):
    """
    Embedding of documents
    
    We rely on two major libraries to embed documents, namely https://github.com/UKPLab/sentence-transformers and https://huggingface.co/transformers/
    """
    # -------------------------------------------------------------------------
    # Constructor function
    # -------------------------------------------------------------------------
    def __init__(self,
                 normalize=True,
                 verbose=True,
                 ):
        self.normalize = normalize
        self.verbose = verbose
        
    # -------------------------------------------------------------------------
    # Class variables
    # -------------------------------------------------------------------------
    STR_USING_SAVED_MODEL = "Using saved model: '{0}'"
    STR_DOWNLOADING_MODEL = "Downloading model: '{0}'"    
    STR_DOCUMENTS_NEED_EMBEDDING_INPUT = "Embeddings from model '{0}' needed"

    # -------------------------------------------------------------------------
    # Private functions
    # -------------------------------------------------------------------------
    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        """
        See example here: https://huggingface.co/sentence-transformers/all-roberta-large-v1
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
        return sum_embeddings / sum_mask

    def _check_embeddings_existence(self, documents, model_name_or_path):
        """Check which documents that need to be embeddings versus which documents that have already been embedded"""
        
        # Check if embeddings exist using this particular method
        if hasattr(self, model_name_or_path+"-embeddings"):
            if self.verbose:
                print(f"Embeddings from model '{model_name_or_path}' found")
            # Load existing embeddings
            documents_embeddings = getattr(self, model_name_or_path+"-embeddings")
            
            # Check which documents have not already been embedded
            documents_not_embedded = []
            for doc in documents:
                if documents_embeddings.get(doc) is None:
                    documents_not_embedded.append(doc)                
        else:
            setattr(self, model_name_or_path+"-embeddings", {})
            documents_not_embedded = documents
        
        return documents_not_embedded

    
    def _extract_existing_embeddings(self, documents, model_name_or_path):
        """Extract those embeddings that have already been embedded"""
        # Extract relevant document embeddings
        embeddings = {k:getattr(self, model_name_or_path+"-embeddings")[k] for k in documents}
        
        return embeddings


    def _embed_documents(self,
                         documents,
                         model_name_or_path="all-roberta-large-v1"):
        # ---------------------------------------------------------------------
        # Check documents have been embedded before
        # ---------------------------------------------------------------------
        documents_not_embedded = self._check_embeddings_existence(documents=documents,
                                                                  model_name_or_path=model_name_or_path)

        if bool(documents_not_embedded):
            # We have assessed that some documents have not been embedded, hence we construct embeddings and save them!
            if self.verbose:
                print(self.STR_DOCUMENTS_NEED_EMBEDDING_INPUT.format(model_name_or_path))
                
            # -----------------------------------------------------------------
            # Check if model is already loaded
            # -----------------------------------------------------------------
            if hasattr(self, model_name_or_path+"-model"):
                if self.verbose:
                    print(self.STR_USING_SAVED_MODEL.format(model_name_or_path))
                    
                pretrained_model = getattr(self, model_name_or_path+"-model")
                tokenizer = getattr(self, model_name_or_path+"-tokenizer")
                
            else:
                if self.verbose:
                    print(self.STR_DOWNLOADING_MODEL.format(model_name_or_path))
    
                try:
                    # Load model via SentenceTransformer
                    pretrained_model = SentenceTransformer(model_name_or_path=model_name_or_path)
                    tokenizer = None
                    self.loaded_via = "SentenceTransformer"
                except:                
                    try:
                        # Load model via HuggingFace
                        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                        self.loaded_via = "HuggingFace"
                    except OSError as err:
                        raise Exception(f"Could not load '{model_name_or_path}'. \nWhile trying, this error occured: \n\n{err}")
                        
                # Set tokenizer and pretrained model as attributes under the model name.
                setattr(self, model_name_or_path+"-model", pretrained_model)
                setattr(self, model_name_or_path+"-tokenizer", tokenizer)        
                
            # -----------------------------------------------------------------
            # Embed
            # -----------------------------------------------------------------
            # Maximum sentence length
            max_seq_length = max([256, max([len(d.split()) for d in documents_not_embedded])])    
            
            if self.loaded_via=="SentenceTransformer":
                            
                # Update maximum sequence length
                pretrained_model.max_seq_length = max([pretrained_model.max_seq_length, max_seq_length])            
    
                # Construct embedding
                raw_embeddings = pretrained_model.encode(documents_not_embedded,
                                                         batch_size=32,
                                                         show_progress_bar=False,
                                                         output_value="sentence_embedding",
                                                         convert_to_numpy=True,
                                                         convert_to_tensor=False,
                                                         device=None,
                                                         normalize_embeddings=True)
                            
            elif self.loaded_via=="HuggingFace":
    
                # Tokenize sentences
                encoded_input = tokenizer(text=documents_not_embedded,
                                          padding=True,
                                          truncation=True,
                                          max_length=max_seq_length,
                                          return_tensors='pt')
            
                # Compute token embeddings
                with torch.no_grad():
                    model_output = pretrained_model(**encoded_input)
            
                # Perform pooling (here, mean pooling)
                raw_embeddings = self._mean_pooling(model_output=model_output, attention_mask=encoded_input['attention_mask'])                
    
                # Convert tensor to numpy array
                raw_embeddings = raw_embeddings.numpy()
    
            # -----------------------------------------------------------------
            # Finalize
            # -----------------------------------------------------------------
                    
            # Pre-allocate
            embeddings = {}
            
            for i,doc in enumerate(documents_not_embedded):
                embeddings[doc] = raw_embeddings[i,:].reshape(-1,)
                
            # Update existing document embeddings
            setattr(self, model_name_or_path+"-embeddings", {**getattr(self, model_name_or_path+"-embeddings"),
                                                             **embeddings}
                    )            
            
            
        # Extract embeddings
        embeddings_out = self._extract_existing_embeddings(documents=documents,
                                                           model_name_or_path=model_name_or_path)
        
        return embeddings_out
    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def show_available_models(self):
        """Print available models via SentenceTransformers"""
        
        # Last last updated
        DATE_LAST_UPDATED = "February 3, 2022"
        
        # Models
        AVAILABLE_MODELS_VIA_SENTENCETRANSFORMERS = ["all-roberta-large-v1",
                                                     "all-mpnet-base-v1",
                                                     "all-mpnet-base-v2"
                                                     "all-MiniLM-L12-v1",
                                                     "all-distilroberta-v1",
                                                     "all-MiniLM-L12-v2",
                                                     "all-MiniLM-L6-v2",
                                                     "all-MiniLM-L6-v1",
                                                     "paraphrase-mpnet-base-v2",
                                                     "multi-qa-mpnet-base-dot-v1",
                                                     "multi-qa-distilbert-dot-v1",
                                                     "multi-qa-mpnet-base-cos-v1",
                                                     "paraphrase-distilroberta-base-v2",
                                                     "paraphrase-TinyBERT-L6-v2",
                                                     "paraphrase-MiniLM-L12-v2",
                                                     "multi-qa-distilbert-cos-v1",
                                                     "paraphrase-multilingual-mpnet-base-v2",
                                                     "paraphrase-MiniLM-L6-v2",
                                                     "paraphrase-albert-small-v2",
                                                     "multi-qa-MiniLM-L6-cos-v1",
                                                     "paraphrase-multilingual-MiniLM-L12-v2",
                                                     "multi-qa-MiniLM-L6-dot-v1",
                                                     "msmarco-bert-base-dot-v5",
                                                     "msmarco-distilbert-base-tas-b",
                                                     "paraphrase-MiniLM-L3-v2",
                                                     "msmarco-distilbert-dot-v5",
                                                     "distiluse-base-multilingual-cased-v1",
                                                     "distiluse-base-multilingual-cased-v2",
                                                     "average_word_embeddings_komninos",
                                                     "average_word_embeddings_glove.6B.300d",
                                                     ]
        
        print(f"""As of out {DATE_LAST_UPDATED}, these models are available through SentenceTransformers:
              
              {AVAILABLE_MODELS_VIA_SENTENCETRANSFORMERS}
              """
              )
        return AVAILABLE_MODELS_VIA_SENTENCETRANSFORMERS


    def embed_documents(self,
                        documents,
                        return_type="df",
                        model_name_or_path="all-roberta-large-v1"):
        RETURN_TYPE_OPT = ["dict", "df"]
        
        # Check documents
        if isinstance(documents, list):
            pass
        elif isinstance(documents, str):
            documents = [documents]
        else:
            raise bg.exceptions.WrongInputException(input_name="documents",
                                                    provided_input=documents,
                                                    allowed_inputs=["str", "list"])
            
        # Embed documents
        embeddings = self._embed_documents(documents=documents,
                                           model_name_or_path=model_name_or_path)
                    
        # Normalize
        if self.normalize:
            if isinstance(embeddings, dict):             
                embeddings = {k: normalize_by_norm(x=v,norm="L2") for k,v in embeddings.items()}
                
        # Convert to df to change column type              
        embeddings = convert_dict_to_df(x=embeddings)
        
        # Enforce columns to be strings
        embeddings.columns = embeddings.columns.astype(str)
                
        if return_type=="df":
            pass
        elif return_type=="dict":
            embeddings = convert_df_to_dict(x=embeddings)    
        else:
            raise bg.exceptions.WrongInputException(input_name="return_type",
                                                    provided_input=return_type,
                                                    allowed_inputs=RETURN_TYPE_OPT)
        
        return embeddings
    




