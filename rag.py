

# rag.py - Retrieval-Augmented Generation Engine

# This module acts as the bridge between the user's raw query and the powerful LLM.
# It uses the vector database to find relevant examples and then constructs a high-quality
# "few-shot" prompt to guide the LLM towards the desired output format and style.

from vectordb import get_category_from_prompt, find_similar_prompts

def get_enhanced_prompt(user_story: str) -> str:
    """
    Creates a RAG-enhanced prompt for the LLM.

    This is the core of the RAG system. It performs a 3-step process:
    1.  **Classify**: Determines the category of the user's query (e.g., 'group_settlement').
    2.  **Retrieve**: Fetches the top 3 most relevant examples from that category in the vector DB.
    3.  **Generate**: Constructs a new, detailed prompt that includes these examples, guiding
        the LLM to produce a precise and correctly formatted response.

    Args:
        user_story: The raw query from the user in Hinglish.

    Returns:
        A string containing the full, enhanced few-shot prompt ready for the LLM.
    """
    # Step 1: User prompt ki category pata karo.
    # Hum 'vectordb' module ka istemal karke user ke prompt ko classify karte hain.
    category = get_category_from_prompt(user_story)
    print(f"✅ Identified Category: '{category}'")

    # Step 2: Uss category se 3 sabse milte-julte (semantic) examples nikalo.
    # Yeh examples LLM ko sahi format mein jawab dene ke liye guide karenge.
    similar_examples = find_similar_prompts(user_story, category, top_k=3)
    print(f"✅ Retrieved {len(similar_examples)} similar examples.")

    # Step 3: LLM ke liye final prompt taiyaar karo.
    
    # Yeh hamara base instruction hai jo LLM ko batata hai ki use kya karna hai.
    final_prompt = """You are an expert financial assistant. Your primary task is to analyze a user's story in Hinglish and provide a clear, step-by-step financial summary in Hindi. Please use the user's currency and values accurately.

"""

    # Agar humein relevant examples mile hain, to unhe prompt mein jodo.
    if similar_examples:
        final_prompt += "Use the following examples to understand the required format and calculation style.\n\n"
        
        # Har example ko ek saaf format mein prompt mein add karo.
        for i, example in enumerate(similar_examples, 1):
            final_prompt += f"--- EXAMPLE {i} ---\n"
            final_prompt += f"User Text: \"{example['user_text']}\"\n"
            final_prompt += f"Your Response:\n{example['model_response']}\n"
            final_prompt += f"--- END EXAMPLE {i} ---\n\n"

    # Examples ke baad, LLM ko user ka actual sawaal do.
    final_prompt += "Now, analyze the following user's story and provide the financial summary in the same way.\n\n"
    final_prompt += "--- FINAL TASK ---\n"
    final_prompt += f"User Text: \"{user_story}\"\n"
    final_prompt += "Your Response:\n"

    return final_prompt