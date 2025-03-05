import asyncio

import phonetics
import unidecode
import re
from nltk.tokenize import word_tokenize
from metaphone import doublemetaphone
from Levenshtein import distance as levenshtein_distance, distance
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from googletrans import Translator
from typing import Dict, List, Tuple, Set, Optional, Any
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer




def normalize_text(title: str, stop_words: Set[str]) -> str:
    """Normalize text by removing accents, special characters, and stop words."""
    # Convert to lowercase
    title = title.lower()
    # Remove accents and special characters
    title = unidecode.unidecode(title)
    # Remove extra whitespace and non-alphanumeric characters
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    # Remove duplicate letters (e.g., 'marrco' -> 'marco')
    title = re.sub(r'(.)\1+', r'\1', title)
    # Tokenize and remove stop words
    tokens = word_tokenize(title)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def get_periodicity_words() -> Set[str]:
    """Get list of periodicity-related words."""
    return {
        'daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'annual',
        'biweekly', 'bimonthly', 'fortnightly', 'evening', 'morning',
        'night', 'afternoon', 'sandhya', 'pratidin', 'masik', 'saptahik'
    }


def check_periodicity_modification(title: str, existing_title: str) -> bool:
    """Check if the new title is just adding periodicity to an existing title."""
    periodicity_words = get_periodicity_words()
    title_words = set(title.lower().split())
    existing_words = set(existing_title.lower().split())

    # Check if the only difference is periodicity words
    diff_words = title_words.symmetric_difference(existing_words)
    return all(word in periodicity_words for word in diff_words)


def phonetic_encoding(title: str) -> Tuple[str, str]:
    """Generate phonetic encodings using Soundex and Metaphone."""
    words = title.split()
    metaphone_codes = []
    soundex_codes = []

    for word in words:
        # Get Soundex code
        soundex = phonetics.soundex(word)
        soundex_codes.append(soundex)

        # Get both primary and secondary Metaphone codes
        primary, secondary = doublemetaphone(word)
        metaphone_codes.append(primary)
        if secondary:  # If there's a secondary code, include it
            metaphone_codes.append(secondary)

    return ' '.join(soundex_codes), ' '.join(metaphone_codes)


def calculate_similarity_percentage(title1: str, title2: str, soundex1: str, soundex2: str,
                                    metaphone1: str, metaphone2: str, has_prefix1: bool,
                                    has_prefix2: bool, has_suffix1: bool, has_suffix2: bool) -> float:
    """Calculate similarity percentage between two titles using multiple metrics."""
    # Calculate Levenshtein similarity for full titles
    lev_distance = levenshtein_distance(title1, title2)
    max_length = max(len(title1), len(title2))
    lev_similarity = (1 - lev_distance / max_length) * 100

    # Calculate word-by-word Levenshtein similarity
    words1 = title1.split()
    words2 = title2.split()
    word_similarities = []

    for i, word1 in enumerate(words1):
        if i < len(words2):
            word2 = words2[i]
            word_distance = levenshtein_distance(word1, word2)
            word_max_length = max(len(word1), len(word2))
            if word_max_length > 0:
                word_similarity = (1 - word_distance / word_max_length) * 100
                word_similarities.append(word_similarity)

    word_by_word_similarity = sum(word_similarities) / len(word_similarities) if word_similarities else 0

    # Calculate phonetic similarity using Soundex
    words1_soundex = set(soundex1.split())
    words2_soundex = set(soundex2.split())
    soundex_matches = words1_soundex.intersection(words2_soundex)
    soundex_similarity = len(soundex_matches) / max(len(words1_soundex), len(words2_soundex)) * 100

    # Calculate phonetic similarity using Metaphone
    words1_metaphone = set(metaphone1.split())
    words2_metaphone = set(metaphone2.split())
    metaphone_matches = words1_metaphone.intersection(words2_metaphone)
    metaphone_similarity = len(metaphone_matches) / max(len(words1_metaphone), len(words2_metaphone)) * 100

    # Calculate individual word phonetic matches
    word_phonetic_similarities = []
    for i, word1 in enumerate(words1):
        if i < len(words2):
            word2 = words2[i]
            # Get phonetic codes for individual words
            w1_soundex = phonetics.soundex(word1)
            w2_soundex = phonetics.soundex(word2)
            w1_metaphone = doublemetaphone(word1)[0]
            w2_metaphone = doublemetaphone(word2)[0]

            # Calculate word-level phonetic similarity
            word_phonetic_sim = 100 if w1_soundex == w2_soundex else 50
            word_phonetic_sim += 100 if w1_metaphone == w2_metaphone else 50
            word_phonetic_sim /= 2
            word_phonetic_similarities.append(word_phonetic_sim)

    word_phonetic_similarity = sum(word_phonetic_similarities) / len(
        word_phonetic_similarities) if word_phonetic_similarities else 0

    # Calculate final phonetic similarity with emphasis on exact matches
    phonetic_similarity = (0.3 * soundex_similarity) + (0.3 * metaphone_similarity) + (0.4 * word_phonetic_similarity)

    # Calculate final similarity with adjusted weights
    final_similarity = (0.2 * lev_similarity) + (0.3 * word_by_word_similarity) + (0.5 * phonetic_similarity)

    # Boost similarity if the words are very similar phonetically
    if word_phonetic_similarity > 80:
        final_similarity = max(final_similarity, word_phonetic_similarity)

    # Boost similarity if both titles share prefixes/suffixes
    if has_prefix1 and has_prefix2 and words1[0].lower() == words2[0].lower():
        final_similarity += 10
    if has_suffix1 and has_suffix2 and words1[-1].lower() == words2[-1].lower():
        final_similarity += 10

    return min(final_similarity, 100)  # Cap at 100%


def check_combination_of_existing_titles(title: str, existing_titles: List[str]) -> Optional[str]:
    """Check if the new title is a combination of existing titles."""
    title_words = set(title.lower().split())

    # Check each pair of existing titles
    for i, title1 in enumerate(existing_titles):
        words1 = set(title1.lower().split())
        if words1.issubset(title_words):
            # Found a title that's part of the new title
            remaining_words = title_words - words1
            if remaining_words:  # If there are remaining words
                # Check if remaining words form another existing title
                remaining_text = ' '.join(sorted(remaining_words))
                for j, title2 in enumerate(existing_titles):
                    if i != j:  # Don't compare with the same title
                        if all(word in remaining_text for word in title2.lower().split()):
                            return f"{title1} + {title2}"
    return None


def calculate_verification_probability(similarity_score: float) -> float:
    """Calculate verification probability based on similarity score."""
    # If similarity is high (>80%), probability is very low
    if similarity_score > 80:
        return 0
    # Linear decrease in probability as similarity increases
    return max(0, 100 - similarity_score)


'''async def check_similar_meanings(title: str, existing_title: str) -> bool:
    translator = Translator()

    # Function to translate only non-English titles
    loop = asyncio.get_event_loop()

    # Translate new and existing titles

    trans_new_title = await loop.run_in_executor(None, lambda: translator.translate(title, dest='en'))
    trans_existing_title = await loop.run_in_executor(None, lambda: translator.translate(existing_title, dest="en"))

    # Function to check phonetic similarity
    def phonetic_match(title1, title2):
        return doublemetaphone(title1)[0] == doublemetaphone(title2)[0]  # Compare primary encodings

    #def levenshtein_distance(title1, title2):
    #    return distance(title1, title2)
    # Check for matches

    if (
            trans_existing_title == trans_new_title  # Exact match
            or phonetic_match(trans_existing_title, trans_new_title)  # Phonetic similarity
            or fuzz.ratio(trans_existing_title, trans_new_title) > 80  # Fuzzy match threshold
            # or levenshtein_distance(trans_existing_title, trans_new_title)>=5
    ):
        return True

    return False
'''
def check_similar_meanings(title: str, existing_title: str) -> bool:
    """Check if titles have similar meanings across languages."""
    # Map of equivalent words in different languages
    translation_map = {
        'daily': {'pratidin', 'dainik'},
        'evening': {'sandhya', 'sayankaal'},
        'morning': {'pratah', 'subah'},
        'news': {'samachar', 'vartha'},
        'weekly': {'saptahik', 'vaaram'},
        'monthly': {'masik', 'maasam'},
    }

    title_words = set(title.lower().split())
    existing_words = set(existing_title.lower().split())

    for eng_word, translations in translation_map.items():
        if (eng_word in title_words and translations.intersection(existing_words)) or \
                (eng_word in existing_words and translations.intersection(title_words)):
            return True

    return False

def check_prefix_suffix(title: str, prefixes: List[str], suffixes: List[str]) -> Tuple[bool, bool]:
    """Check if the title starts with any prefix or ends with any suffix."""
    words = title.split()
    first_word = words[0] if words else ""
    last_word = words[-1] if words else ""
    has_prefix = any(first_word.lower() == prefix for prefix in prefixes)
    has_suffix = any(last_word.lower() == suffix for suffix in suffixes)
    return has_prefix, has_suffix


def contains_disallowed_words(title: str, disallowed_words: Set[str]) -> Set[str]:
    """Check if the title contains any disallowed words."""
    words = set(title.split())
    disallowed_found = words.intersection(disallowed_words)
    return disallowed_found


def check_similar_titles_meaning(title1: str, title2: str, threshold=0.7) -> bool:
    """Check if two titles are similar using a transformer model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embedding1 = model.encode(title1)
    embedding2 = model.encode(title2)
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0] >= threshold


def encode_title(title: str, tokenizer, model) -> np.ndarray:
    """Encode a title using a transformer model."""
    inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

