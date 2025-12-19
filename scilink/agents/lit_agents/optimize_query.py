from typing import Any

def optimize_search_query(objective: str, 
                          search_intent: str, 
                          model: Any, 
                          generation_config: Any) -> str:
    """
    Translates a raw, messy user objective into a clean, targeted search query.
    """
    prompt = f"""
    You are a Research Librarian.
    
    **USER INPUT:** "{objective}"
    **SEARCH INTENT:** {search_intent}
    
    **TASK:** Convert this input (which may refer to specific local files, results, or objectives) into a **General Scientific Search Query** suitable for a database like Google Scholar.
    
    **RULES:**
    1. Remove references to "provided data", "this spreadsheet", "my results".
    2. Extract the core scientific topic.
    3. Return ONLY the query string.
    """
    
    try:
        response = model.generate_content([prompt], generation_config=generation_config)
        
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