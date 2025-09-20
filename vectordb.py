# vectordb.py - Vector Database and Semantic Search Engine
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_FILE_PATH = "hissab_vector_db.pkl"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Global Variables ---
# Load the embedding model once and store it in memory
embedding_model = SentenceTransformer(MODEL_NAME)
# The database will be loaded into this DataFrame
hissab_db = None

# --- Initial Data Prompts (10 Refined Categories) ---
INITIAL_PROMPTS = [
    {
        "category": "personal_expense_tracking",
        "user_text": "Aaj 30 rupaye bus ke lage, 250 ka khana khaya, aur 500 ki ek shirt kharidi.",
        "model_response": "Aapke aaj ke kul kharch is prakaar hain:\n- Bus: ₹30\n- Khana: ₹250\n- Shirt: ₹500\n**Kul Kharch: ₹830**"
    },
    {
        "category": "group_settlement",
        "user_text": "Hum 3 dost, main, Rohit aur Suman, Goa gaye. Maine hotel ke 6000 diye, Rohit ne khaane ke 3000 kharch kiye. Suman ne kuch nahi diya.",
        "model_response": "**Trip ka Hisaab:**\n- **Kul Kharch:** ₹9,000 (₹6000 Hotel + ₹3000 Khaana)\n- **Log:** 3\n- **Prati Vyakti Hissa:** ₹3,000\n**Settlement:**\n- Aapne ₹3,000 extra diye hain.\n- Rohit ne apna hissa de diya hai.\n- **Isliye, Suman ko aapko ₹3,000 dene hain.**"
    },
    {
        "category": "monthly_budget_and_savings",
        "user_text": "Meri salary 50000 hai. 15000 ghar ka kiraya, 3000 bijli paani, 10000 ration aur 5000 aane jaane mein lag jaata hai. Kitni bachat hoti hai?",
        "model_response": "**Aapka Maheene ka Hisaab:**\n- **Kul Aamdani (Salary):** ₹50,000\n- **Kul Kharch:** ₹33,000 (₹15000 Kiraya + ₹3000 Bill + ₹10000 Ration + ₹5000 Aana-Jaana)\n- **Isliye, aapki kul bachat ₹17,000 hai.**"
    },
    {
        "category": "price_comparison",
        "user_text": "Ek dukaan par Samsung ka phone 25000 ka hai aur wahi phone doosri dukaan par 23500 ka mil raha hai. Kaunsa sasta hai?",
        "model_response": "**Cheezon ki Tulna:**\n- Dusri dukaan wala phone sasta hai.\n- **Dono ke beech ₹1,500 ka antar hai.**"
    },
    {
        "category": "lending_and_borrowing",
        "user_text": "Maine apne dost Aman ko 2000 rupaye udhaar diye the, usne mujhe 500 lauta diye hain. Ab kitne paise lene baaki hain?",
        "model_response": "**Udhaari ka Hisaab:**\n- **Kul Udhaar:** ₹2,000\n- **Vaapas Mile:** ₹500\n- **Isliye, aapko Aman se abhi ₹1,500 aur lene hain.**"
    },
    {
        "category": "investment_and_profit",
        "user_text": "Maine 50,000 rupaye share market mein lagaye the, aur ab unki value 65,000 ho gayi hai. Mujhe kitna munafa hua?",
        "model_response": "**Nivesh ka Hisaab:**\n- **Aapka Nivesh:** ₹50,000\n- **Abhi ka Value:** ₹65,000\n- **Isliye, aapko kul ₹15,000 ka munafa hua hai.**"
    },
    {
        "category": "loan_and_emi",
        "user_text": "Mera 2 lakh ka personal loan hai aur har mahine 5000 ki EMI jaati hai. 1 saal mein main kitna chuka dunga?",
        "model_response": "**Loan ka Hisaab:**\n- **Har Mahine ki EMI:** ₹5,000\n- **Ek Saal (12 Mahine) mein Kul Bhugtaan:** ₹60,000 (5000 x 12)\n- **Isliye, aap ek saal mein ₹60,000 chuka denge.**"
    },
    {
        "category": "income_and_balance",
        "user_text": "Mere account mein 10000 the, mujhe 25000 salary mili. Uske baad maine 2000 ka bill bhara. Ab kitne bache?",
        "model_response": "**Account ka Hisaab:**\n- **Shuruaati Balance:** ₹10,000\n- **Salary Aayi:** + ₹25,000\n- **Bill Bhara:** - ₹2,000\n- **Aapka abhi ka balance ₹33,000 hai.**"
    },
    {
        "category": "discount_and_offers",
        "user_text": "Ek jacket 4000 ki hai aur us par 20% ka discount hai. Mujhe kitne paise dene honge?",
        "model_response": "**Discount ka Hisaab:**\n- **Jacket ka Daam:** ₹4,000\n- **Discount (20%):** ₹800 (4000 ka 20%)\n- **Isliye, aapko ₹3,200 dene honge.**"
    },
    {
        "category": "salary_calculation",
        "user_text": "Main din ke 800 rupaye kamata hoon. Is mahine maine 25 din kaam kiya. Meri is mahine ki salary kitni hui?",
        "model_response": "**Salary ka Hisaab:**\n- **Ek Din ki Kamai:** ₹800\n- **Kul Kaam ke Din:** 25\n- **Isliye, aapki is mahine ki salary ₹20,000 hui (800 x 25).**"
    }
]

def _initialize_database():
    """
    Creates the initial vector database from the hardcoded prompts,
    generates embeddings, and saves it to a file.
    """
    print("Vector DB banaya ja raha hai...")
    df = pd.DataFrame(INITIAL_PROMPTS)
    # Generate embeddings for each user_text
    embeddings = embedding_model.encode(df['user_text'].tolist(), show_progress_bar=True)
    df['embedding'] = list(embeddings)
    df.to_pickle(DB_FILE_PATH)
    print(f"Vector DB '{DB_FILE_PATH}' mein save ho gaya hai.")
    return df

def setup_vector_db():
    """
    Loads the vector database from the file if it exists, otherwise initializes it.
    This is called once when the main app starts.
    """
    global hissab_db
    if os.path.exists(DB_FILE_PATH):
        print(f"Pehle se bani hui Vector DB '{DB_FILE_PATH}' se load ho rahi hai...")
        hissab_db = pd.read_pickle(DB_FILE_PATH)
    else:
        print("Pehli baar setup kiya ja raha hai...")
        hissab_db = _initialize_database()

def get_category_from_prompt(user_prompt: str) -> str:
    """
    Uses the Gemini model to classify the user's prompt into one of the predefined categories.
    """
    if not user_prompt:
        return "unknown"
        
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Error: Google API Key missing.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Get a unique list of categories from our DB
    category_list = hissab_db['category'].unique().tolist()
    
    # Create a specific prompt for the classification task
    classification_prompt = f"""
    You are a financial query classification expert. Your task is to classify the following user query into one of the given categories.
    Only return the category name and nothing else.

    Available Categories: {', '.join(category_list)}

    User Query: "{user_prompt}"

    Category:
    """
    
    try:
        response = model.generate_content(classification_prompt)
        # Clean up the response to get only the category name
        category = response.text.strip().lower()
        # Ensure the model returns a valid category
        if category in category_list:
            return category
        else:
            # Fallback if the model returns something unexpected
            return "personal_expense_tracking"
    except Exception as e:
        print(f"Category pata karte samay error aaya: {e}")
        # Default to a general category on error
        return "personal_expense_tracking"


def find_similar_prompts(user_prompt: str, category: str, top_k: int = 3) -> list:
    """
    Finds the most similar prompts from the database within a specific category.
    """
    global hissab_db
    if hissab_db is None or hissab_db.empty:
        return []

    # 1. Filter the database for the given category
    category_df = hissab_db[hissab_db['category'] == category]
    if category_df.empty:
        return []

    # 2. Generate embedding for the new user prompt
    user_embedding = embedding_model.encode([user_prompt])[0]

    # 3. Calculate cosine similarity
    all_embeddings = np.array(category_df['embedding'].tolist())
    similarities = cosine_similarity([user_embedding], all_embeddings)[0]

    # 4. Get the top_k most similar prompts
    # We use argsort to get indices of the most similar items
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    similar_examples = category_df.iloc[top_indices][['user_text', 'model_response']].to_dict(orient='records')
    
    return similar_examples

def add_user_prompt_to_db(user_prompt: str):
    """
    Adds a new user prompt to the in-memory database and saves it back to the file.
    This allows the DB to grow and improve over time.
    """
    global hissab_db
    print(f"Naya prompt DB mein add kiya ja raha hai: '{user_prompt}'")
    
    # To add a new prompt, we must first classify it
    category = get_category_from_prompt(user_prompt)
    embedding = embedding_model.encode([user_prompt])[0]
    
    # We add it without a model_response for now, as it's just for future semantic matching
    new_entry = pd.DataFrame([{'category': category, 'user_text': user_prompt, 'embedding': embedding, 'model_response': ''}])
    
    hissab_db = pd.concat([hissab_db, new_entry], ignore_index=True)
    
    # Save the updated database to the file for persistence
    hissab_db.to_pickle(DB_FILE_PATH)
    print("DB update ho gaya hai.")