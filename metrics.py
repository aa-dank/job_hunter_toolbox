import re
import json
import math
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from sklearn.metrics import pairwise
# from sentence_transformers import SentenceTransformer  # Imported conditionally when needed
from typing import Any, Dict, Optional, Union, List

class ScoringMethodType(Enum):
    """Enumeration of available scoring method types."""
    SEMANTIC = "semantic"
    LEXICAL = "lexical" 
    STATISTICAL = "statistical"
    HYBRID = "hybrid"

@dataclass
class ScoringStrategy(ABC):
    """
    Abstract base class for resume feature scoring strategies.
    
    This class defines the interface for different scoring methods used to evaluate
    the relevance of resume content against job descriptions.
    """
    
    # Core attributes
    name: str
    description: str
    method_type: ScoringMethodType
    
    # Scoring guidance
    score_range: tuple[float, float] = (0.0, 1.0)
    higher_is_better: bool = True
    
    # Usage instructions for LLM
    llm_usage_instructions: str = ""
    
    # Configuration
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        
        # Set default LLM instructions if not provided
        if not self.llm_usage_instructions:
            direction = "higher" if self.higher_is_better else "lower"
            self.llm_usage_instructions = (
                f"Each item includes a relevance score in format [relevance: X.XX] "
                f"ranging from {self.score_range[0]:.1f} to {self.score_range[1]:.1f}. "
                f"{direction.capitalize()} scores indicate stronger relevance to the job. "
                f"Prioritize items with {direction} scores and remove the score notation in your output."
            )
    
    @abstractmethod
    def calculate_score(self, reference_text: str, candidate_text: str, **kwargs) -> float:
        """
        Calculate similarity/relevance score between two texts.
        
        Args:
            reference_text: The reference text (e.g., job description)
            candidate_text: The candidate text (e.g., resume bullet point)
            **kwargs: Additional arguments specific to the scoring method
            
        Returns:
            float: Similarity score within the defined range
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def batch_score(self, reference_text: str, candidate_texts: List[str], **kwargs) -> List[float]:
        """
        Score multiple candidate texts against a single reference.
        
        Default implementation uses individual scoring, but can be overridden
        for more efficient batch processing.
        """
        return [self.calculate_score(reference_text, text, **kwargs) for text in candidate_texts]
    
    def format_for_llm(self, text: str, score: float) -> str:
        """Format text with score for LLM consumption."""
        return f"{text} [relevance: {score:.2f}]"
    
    def get_prompt_instructions(self) -> str:
        """Get instructions to append to LLM prompts."""
        return self.llm_usage_instructions


class SentenceTransformerStrategy(ScoringStrategy):
    """Scoring strategy using SentenceTransformer embeddings."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 name: str = "SentenceTransformer Semantic Similarity",
                 description: str = "Uses transformer-based embeddings for semantic similarity"):
        
        super().__init__(
            name=name,
            description=description,
            method_type=ScoringMethodType.SEMANTIC,
            llm_usage_instructions=(
                "Measures semantic similarity using embeddings from Sentence Transformers and cosine similarity. "
                "Each point is tagged with [relevance: X.XX] within [0.0,1.0], where higher values mean greater relevance. "
                "Use these scores to prioritize the most relevant items in the resume and omit the score notation in your final output."
            ),
            config={"model_name": model_name}
        )
        
        self._model = None
    
    @property
    def model(self):
        """Lazy loading of the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config["model_name"])
            except ImportError:
                raise ImportError("sentence-transformers is required for SentenceTransformerStrategy. Install it with: uv add sentence-transformers")
        return self._model
    
    def calculate_score(self, reference_text: str, candidate_text: str, **kwargs) -> float:
        """Calculate semantic similarity using sentence transformers."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Get embeddings
        emb_ref, emb_cand = self.model.encode([reference_text, candidate_text])
        sim = cosine_similarity([emb_ref], [emb_cand])[0][0]
        
        # Clamp into [score_range]
        score = float(max(self.score_range[0], min(sim, self.score_range[1])))
        return score

    def batch_score(self, reference_text: str, candidate_texts: list[str], **kwargs) -> list[float]:
        """Efficient batch scoring for multiple candidates."""
        from sklearn.metrics.pairwise import cosine_similarity

        if not candidate_texts:
            return []

        # Encode all at once
        embeddings = self.model.encode([reference_text] + candidate_texts)
        ref_emb = embeddings[0:1]
        cand_embs = embeddings[1:]
        sims = cosine_similarity(ref_emb, cand_embs)[0]
        
        # Clamp each into [score_range]
        return [float(max(self.score_range[0], min(s, self.score_range[1]))) for s in sims]
        

def remove_urls(list_of_strings):
    """Removes strings containing URLs from a list using regular expressions."""
    filtered_list = [string for string in list_of_strings if not re.search(r"https?://\S+", string)]
    return filtered_list

def overlap_coefficient(document1: str, document2: str) -> float:
    """Calculate the overlap coefficient between two documents.

    The overlap coefficient is a measure of the overlap between two sets, 
    and is defined as the size of the intersection divided by the smaller 
    of the size of the two sets.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The overlap coefficient between the two documents.
    """    
    # List the unique words in a document
    words_in_document1 = set(normalize_text(document1))
    words_in_document2 = set(normalize_text(document2))

    # Find the intersection of words list of document1 & document2
    intersection = words_in_document1.intersection(words_in_document2)

    # Calculate overlap coefficient
    try:
        overlap_coefficient = float(len(intersection)) / min(len(words_in_document1), len(words_in_document2))
    except ZeroDivisionError:
        overlap_coefficient = 0.0

    return overlap_coefficient
    
def jaccard_similarity(document1: str, document2: str) -> float:
    """Calculate the Jaccard similarity between two documents.

    The Jaccard similarity is a measure of the similarity between two sets, 
    and is defined as the size of the intersection divided by the size of 
    the union of the two sets.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The Jaccard similarity between the two documents.
    """    
    # List the unique words in a document
    words_in_document1 = set(normalize_text(document1))
    words_in_document2 = set(normalize_text(document2))

    # Find the intersection of words list of document1 & document2
    intersection = words_in_document1.intersection(words_in_document2)

    # Find the union of words list of document1 & document2
    union = words_in_document1.union(words_in_document2)
        
    # Calculate Jaccard similarity score 
    try:
        jaccard_similarity = float(len(intersection)) / len(union)
    except ZeroDivisionError:
        jaccard_similarity = 0.0

    return jaccard_similarity

def cosine_similarity(document1: str, document2: str) -> float:
    """Calculate the cosine similarity between two documents.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.

    Returns:
        float: The cosine similarity between the two documents.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the documents into TF-IDF vectors
    vectors = vectorizer.fit_transform([document1, document2])

    cosine_similarity_score = pairwise.cosine_similarity(vectors[0], vectors[1])
    # Calculate the cosine similarity between the two vectors
    # cosine_similarity = np.dot(vectors[0], vectors[1].T) / (np.linalg.norm(vectors[0].toarray()) * np.linalg.norm(vectors[1].toarray()))

    return cosine_similarity_score.item()

def vector_embedding_similarity(llm, document1: str, document2: str) -> float:
    """Calculate similarity between two documents using vector embeddings from a language model.

    This function converts JSON string documents into key-value pairs, gets embeddings using the provided
    language model, and calculates the cosine similarity between the embeddings.

    Args:
        llm: The language model used to generate embeddings.
        document1 (str): The first document as a JSON string.
        document2 (str): The second document as a JSON string.

    Returns:
        float: The mean cosine similarity between the document embeddings.
    """
    
    def key_value_chunking(data, prefix=""):
        """Chunk a dictionary or list into key-value pairs.

        Args:
            data (dict or list): The data to chunk.
            prefix (str, optional): The prefix to use for the keys. Defaults to "".

        Returns:
            A list of strings representing the chunked key-value pairs.
        """
        chunks = []
        stop_needed = lambda value: '.' if not isinstance(value, (str, int, float, bool, list)) else ''
        
        if isinstance(data, dict):
            for key, value in data.items():
                if value is not None:
                    chunks.extend(key_value_chunking(value, prefix=f"{prefix}{key}{stop_needed(value)}"))
        elif isinstance(data, list):
            for index, value in enumerate(data):
                if value is not None:
                    chunks.extend(key_value_chunking(value, prefix=f"{prefix}_{index}{stop_needed(value)}"))
        else:
            if data is not None:
                chunks.append(f"{prefix}: {data}")
        
        return chunks
    document1 = key_value_chunking(json.loads(document1))
    document2 = key_value_chunking(json.loads(document2))
    
    emb_1 = llm.get_embedding(document1, task_type="retrieval_query")
    emb_2 = llm.get_embedding(document2, task_type="retrieval_query")

    df1 = pd.DataFrame(emb_1.embedding.to_list())
    df2 = pd.DataFrame(emb_2.embedding.to_list())

    emb_sem = pairwise.cosine_similarity(df1, df2)

    return emb_sem.mean()

def normalize_text(text: str) -> list:
    """Normalize the input text.

    This function tokenizes the text, removes stopwords and punctuations, 
    and applies stemming.

    Args:
        text (str): The text to normalize.

    Returns:
        list: The list of normalized words.
    """    
    # Import NLTK libraries only when needed
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    
    # Download NLTK data if needed
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Step 1: Tokenization
    words = word_tokenize(text)

    # Step 2: Data Cleaning - Remove Stopwords and Punctuations 
    words = [re.sub('[^a-zA-Z]', '', word).lower() for word in words]

    # Step 3: Remove empty tokens
    words = [word for word in words if len(word)] 

    # Step 4: Remove Stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Step 5: Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    #STEP 3 : LEMMATIZATION
    # lemmatizer=WordNetLemmatizer()
    # words=[lemmatizer.lemmatize(word) for word in words]
    
    return words

def sentence_transformer_similarity(document1: str, document2: str, model = None) -> float:
    """Calculate the cosine similarity between two documents using Sentence Transformers.

    Args:
        document1 (str): The first document.
        document2 (str): The second document.
        model: SentenceTransformer model instance, if None will create default model

    Returns:
        float: The cosine similarity between the two documents.
    """
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError("sentence-transformers is required for this function. Install it with: uv add sentence-transformers")

    # Encode the documents
    embeddings = model.encode([document1, document2])

    # Calculate cosine similarity
    cosine_similarity_score = pairwise.cosine_similarity([embeddings[0]], [embeddings[1]])

    return cosine_similarity_score.item()