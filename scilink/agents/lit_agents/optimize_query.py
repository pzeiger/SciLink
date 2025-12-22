from typing import Any

def optimize_search_query(objective: str, 
                          model: Any, 
                          ) -> str:
    """
    Translates a raw, messy user objective into a clean, targeted search query.
    """
    prompt = f"""
    You are a Research Librarian.
    
    **USER INPUT:** "{objective}"
    
    **TASK:** Convert this input (which may refer to specific local files, results, or objectives) into a **General Scientific Search Query** suitable for a database like Google Scholar.
    
    **RULES:**
    1. REMOVE LOCAL CONTEXT: Strip out references to specific files, provided datasets, or user actions (e.g., "analyze this spreadsheet", "my results", "provided data", "the attached file").
    2. PRESERVE SCIENTIFIC NOUNS: Preserve specific chemical names, material sources (e.g., "Produced Water", "Lithium-Ion Batteries"), and key analytes.
    3. STANDALONE: The output must make sense to a search engine that knows nothing about the user's computer.
    4. Return ONLY the query string.
    """
    
    try:
        response = model.generate_content([prompt])
        
        # Robust extraction
        if hasattr(response, 'text'):
            query = response.text.strip()
        elif hasattr(response, 'parts'):
            query = response.parts[0].text.strip()
        else:
            query = str(response).strip()

        # Clean artifacts
        query = query.replace('"', '').replace("Search Query:", "").strip()
        print(f"  - 🧠 Query Optimized: '{query}'")
        return query
        
    except Exception as e:
        print(f"  - ⚠️ Query optimization failed: {e}. Using raw input.")
        return objective