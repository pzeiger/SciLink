import numpy as np
import faiss
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper_embeddings import OpenAIAsEmbeddingModel
from ...wrappers.litellm_wrapper import LiteLLMEmbeddingModel


from ._deprecation import normalize_params

from openai import RateLimitError


class KnowledgeBase:
    """
    Handles embedding, retrieval, and repository structure mapping.
    Supports both Google and OpenAI-compatible (e.g., incubator) embedding models.

    Args:
        api_key: API key for the embedding provider.
        embedding_model: Name of the embedding model.
        base_url: Base URL for internal proxy endpoint.
        use_litellm: If True and base_url is None, use LiteLLM.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        base_url: Optional[str] = None,
        use_litellm: bool = False,
        # Deprecated parameters
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
   
        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="KnowledgeBase"
        )
        
        self.embedding_model_name = embedding_model
        
        # Initialize embedding client
        if base_url:
            logging.info(f"🏛️ KnowledgeBase using internal proxy for embeddings")
            self.embedding_client = OpenAIAsEmbeddingModel(
                model=embedding_model,
                api_key=api_key,
                base_url=base_url
            )
        elif use_litellm:
            logging.info(f"🌐 KnowledgeBase using LiteLLM for embeddings: {embedding_model}")
            self.embedding_client = LiteLLMEmbeddingModel(
                model=embedding_model,
                api_key=api_key
            )
        else:
            logging.info(f"🔷 KnowledgeBase using OpenAI client for embeddings")
            self.embedding_client = OpenAIAsEmbeddingModel(
                model=embedding_model,
                api_key=api_key
            )
            
        self.index = None
        self.chunks = []
        self.sources: List[str | Dict[str, str]] = []
        
        # Registry for Repo Maps: {'repo_name': 'tree_structure_string'}
        # This stores the visual directory trees for any repo you ingest.
        self.repo_maps: Dict[str, str] = {}

    def build(self, chunks: List[Dict[str, any]], batch_size: int = 100):
        """
        Processes a list of text chunks, generates embeddings in batches, 
        and builds the vector index.
        """
        if not chunks:
            print("⚠️  KnowledgeBase build skipped: No chunks provided.")
            return

        self.chunks.extend(chunks)
        texts_to_embed = [chunk['text'] for chunk in chunks]
        all_embeddings = []
        
        print(f"  - Generating embeddings for {len(texts_to_embed)} chunks using '{self.embedding_model_name}'...")
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            
            max_retries = 3
            delay = 5 # seconds
            for attempt in range(max_retries):
                try:
                    response = self.embedding_client.embed_content(
                        model=self.embedding_model_name,
                        content=batch_texts,
                        task_type="RETRIEVAL_DOCUMENT" # Ignored by OpenAI wrapper, used by Google
                    )
                    all_embeddings.extend(response['embedding'])
                    print(f"    - Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
                    time.sleep(1) # Small delay to respect API rate limits
                    break # Success
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        print(f"    - ⚠️  Rate limit hit during build. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2 # Exponential backoff
                    else:
                        print(f"    - ❌ Rate limit hit on final attempt. Build failed.")
                        raise e 
                except Exception as e:
                    print(f"    - ❌ Error embedding batch {i//batch_size + 1}: {e}")
                    raise e

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        dimension = embeddings_np.shape[1]

        if self.index is None: 
            print("  - Building FAISS vector index...")
            self.index = faiss.IndexFlatL2(dimension)
        else:
            print("  - Appending to existing FAISS vector index...")

        self.index.add(embeddings_np)
        print("  - ✅ Knowledge base built successfully.")

    def save(self, index_path: str, chunks_path: str, repo_map_path: str = None, sources_path: str = None):
        """Saves the FAISS index, text chunks, and optionally the repo maps to disk."""
        if self.index:
            faiss.write_index(self.index, index_path)
            print(f"  - FAISS index saved to {index_path}")
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2)
            print(f"  - Chunks saved to {chunks_path}")

        with open(sources_path, 'w', encoding='utf-8') as f:
            json.dump(self.sources, f, indent=2)
            print(f"  - Sources saved to {sources_path}")

        # Save Repo Maps Registry
        if repo_map_path and self.repo_maps:
            try:
                with open(repo_map_path, 'w', encoding='utf-8') as f:
                    json.dump(self.repo_maps, f, indent=2)
                print(f"  - Repo maps registry saved to {repo_map_path}")
            except Exception as e:
                print(f"  - ❌ Error saving repo maps: {e}")

    def load(self, index_path: str, chunks_path: str, repo_map_path: str = None, sources_path: str = None) -> bool:
        """Loads a pre-built FAISS index, chunks, and repo maps from disk."""
        index_file = Path(index_path)
        chunks_file = Path(chunks_path)
        sources_file = Path(sources_path)

        if not index_file.exists() or not chunks_file.exists() or not sources_file.exists() :
            print("  - ⚠️  Cannot load: Index or chunks or sources file missing.")
            return False
            
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            with open(sources_file, 'r', encoding='utf-8') as f:
                self.sources = json.load(f)
                
            # Load Repo Maps if path provided and file exists
            if repo_map_path and Path(repo_map_path).exists():
                try:
                    with open(repo_map_path, 'r', encoding='utf-8') as f:
                        self.repo_maps = json.load(f)
                    print(f"    - Loaded maps for repos: {list(self.repo_maps.keys())}")
                except Exception as e:
                    print(f"    - ⚠️ Error loading repo maps file: {e}")
            
            print(f"  - ✅ Successfully loaded {len(self.chunks)} chunks and index with {self.index.ntotal} vectors from {len(self.sources)} sources.")
            return True
        except Exception as e:
            print(f"  - ❌ Error loading knowledge base: {e}")
            self.index = None
            self.chunks = []
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks for a given query.
        """
        if not self.index:
            print("⚠️  Cannot retrieve: Knowledge base has not been built.")
            return []
            
        print(f"  - Retrieving top {top_k} most relevant chunks for query: '{query[:80]}...'")

        max_retries = 3
        delay = 5 # seconds
        response = None
        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embed_content(
                    model=self.embedding_model_name,
                    content=query,
                    task_type="RETRIEVAL_QUERY" # Ignored by OpenAI wrapper, used by Google
                )
                break # Success
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"    - ⚠️  Rate limit hit embedding query. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
                else:
                    print(f"    - ❌ Rate limit hit on final attempt. Retrieval failed.")
                    raise e # Re-raise the exception if all retries fail
            except Exception as e:
                print(f"    - ❌ Error embedding query: {e}")
                raise e
        
        if response is None:
            print("    - ❌ Retrieval failed after retries.")
            return []

        query_embedding = np.array([response['embedding']], dtype=np.float32)

        if query_embedding.ndim == 3:
            query_embedding = np.squeeze(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve valid chunks (filtering out potential index errors)
        retrieved_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        print(f"  - ✅ Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks

    def get_relevant_maps(self, retrieved_chunks: List[Dict]) -> str:
        """
        Dynamic Context Injection:
        Looks at the retrieved chunks, finds which repos they belong to (via 'repo_name' metadata),
        and returns a combined string of ONLY the relevant repo maps.
        """
        relevant_repos = set()
        for chunk in retrieved_chunks:
            # We ensure chunks have this metadata field in planning_agent.py
            repo_name = chunk['metadata'].get('repo_name')
            if repo_name and repo_name in self.repo_maps:
                relevant_repos.add(repo_name)
        
        if not relevant_repos:
            return ""

        combined_map = ""
        for repo in relevant_repos:
            combined_map += f"\n--- DIRECTORY STRUCTURE FOR REPO: {repo} ---\n"
            combined_map += self.repo_maps[repo]
            combined_map += "\n"
            
        return combined_map
       
    def source_difference(self, new_sources: List[str | Dict[str, str]]) -> List[str | Dict[str, str]]:
        """Returns the subset of new sources which are not present in the existing sources."""
        
        if not new_sources:
            return []

        # Check if the new sources are dictionaries
        contains_dict = any(isinstance(item, dict) for item in new_sources)
        
        if contains_dict:
            # 1. Convert new sources to tuples for set comparison
            new_sources_tuple = {tuple(sorted(d.items())) for d in new_sources if isinstance(d, dict)}
            
            # 2. Filter existing sources to ONLY check dictionaries
            old_sources_tuple = {
                tuple(sorted(d.items())) 
                for d in self.sources 
                if isinstance(d, dict)
            }
            
            # 3. Calculate difference and convert back to dicts
            difference_tuples = new_sources_tuple - old_sources_tuple
            source_difference = [dict(t) for t in difference_tuples]
            
        else:
            # 1. Filter existing sources to only check strings
            existing_strings = {s for s in self.sources if isinstance(s, str)}
            
            # 2. Calculate difference
            source_difference = list(set(new_sources) - existing_strings)
        
        # Update history
        self.sources.extend(source_difference)
        return source_difference