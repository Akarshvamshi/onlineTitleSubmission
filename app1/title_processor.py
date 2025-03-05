import asyncio

import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from .utils import (
    normalize_text,
    phonetic_encoding,
    check_prefix_suffix,
    contains_disallowed_words,
    calculate_similarity_percentage,
    check_similar_titles_meaning,
    check_combination_of_existing_titles,
    check_periodicity_modification,
    check_similar_meanings,
    calculate_verification_probability,
)

"""nltk.download('punkt')
nltk.download('stopwords')"""


class TitleProcessor:
    def __init__(self):
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))

        # Common prefixes and suffixes in Indian news titles
        self.prefixes = [
            'the', 'india', 'samachar', 'news', 'times', 'daily', 'weekly',
            'dainik', 'hindu', 'express', 'aaj', 'today'
        ]
        self.suffixes = [
            'news', 'samachar', 'report', 'edition', 'times', 'today', 'india',
            'express', 'patrika', 'post', 'herald', 'gazette'
        ]

        # Words that should not be allowed in titles
        self.disallowed_words = {
            'police', 'crime', 'corruption', 'cbi', 'cid', 'army',
            'military', 'intelligence', 'defence', 'weapon', 'terror',
            'terrorist', 'naxal', 'maoist'
        }

        # Load multilingual BERT model and tokenizer for semantic similarity
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def process_title(self, title, existing_titles):
        """
        Process a new title submission and check it against existing titles.
        Returns a dictionary with the status, reason, and verification details.
        """
        # Step 1: Normalize the new title
        normalized_title = normalize_text(title, self.stop_words)
        soundex_code, metaphone_code = phonetic_encoding(normalized_title)
        has_prefix, has_suffix = check_prefix_suffix(normalized_title, self.prefixes, self.suffixes)

        # Initialize result with basic information
        result = {
            "normalized": normalized_title,
            "soundex": soundex_code,
            "metaphone": metaphone_code,
            "has_prefix": has_prefix,
            "has_suffix": has_suffix,
            "verification_probability": 100.0  # Start with 100% probability
        }

        # Step 2: Check for disallowed words
        disallowed_found = contains_disallowed_words(normalized_title, self.disallowed_words)
        if disallowed_found:
            result.update({
                "status": "rejected",
                "reason": f"Title contains disallowed words: {', '.join(disallowed_found)}",
                "verification_probability": 0.0
            })
            return result

        # Step 3: Check for combinations of existing titles
        combined_titles = check_combination_of_existing_titles(normalized_title, existing_titles)
        if combined_titles:
            result.update({
                "status": "rejected",
                "reason": f"Title appears to be a combination of existing titles: {combined_titles}",
                "verification_probability": 0.0
            })
            return result

        # Step 4: Check against existing titles
        max_similarity = 0.0
        similar_title = None

        for existing_title in existing_titles:
            # Normalize existing title
            normalized_existing = normalize_text(existing_title, self.stop_words)

            # Skip if exactly the same
            if normalized_title == normalized_existing:
                result.update({
                    "status": "rejected",
                    "reason": f"Title is identical to existing title: {existing_title}",
                    "verification_probability": 0.0
                })
                return result

            # Check for periodicity modification
            if check_periodicity_modification(normalized_title, normalized_existing):
                result.update({
                    "status": "rejected",
                    "reason": f"Title is just adding periodicity to existing title: {existing_title}",
                    "verification_probability": 0.0
                })
                return result

            # Check for similar meanings across languages
            is_true = (check_similar_meanings(normalized_title, normalized_existing))
            i = is_true
            if is_true:
                result.update({
                    "status": "rejected",
                    "reason": f"Title has similar meaning to existing title: {existing_title}",
                    "verification_probability": 0.0
                })
                return result

            if check_similar_titles_meaning(normalized_title, normalized_existing):
                result.update({
                    "status": "rejected",
                    "reason": f"Title has similar meaning to existing title: {existing_title}",
                    "verification_probability": 0.0
                })
                return result

            # Calculate full similarity
            existing_soundex, existing_metaphone = phonetic_encoding(normalized_existing)
            existing_prefix, existing_suffix = check_prefix_suffix(normalized_existing, self.prefixes, self.suffixes)

            similarity = calculate_similarity_percentage(
                normalized_title, normalized_existing,
                soundex_code, existing_soundex,
                metaphone_code, existing_metaphone,
                has_prefix, existing_prefix,
                has_suffix, existing_suffix
            )

            # Track maximum similarity
            if similarity > max_similarity:
                max_similarity = similarity
                similar_title = existing_title

        # Calculate verification probability based on maximum similarity
        verification_prob = calculate_verification_probability(max_similarity)
        result["verification_probability"] = verification_prob

        # Reject if similarity is too high
        if max_similarity >= 60:  # Threshold for similarity
            result.update({
                "status": "rejected",
                "reason": f"Title is too similar to existing title: {similar_title} (Similarity: {max_similarity:.1f}%)"
            })
            return result

        # If we get here, the title is accepted
        result["status"] = "accepted"
        if max_similarity > 0:
            result[
                "warning"] = f"Note: Most similar existing title is '{similar_title}' with {max_similarity:.1f}% similarity"

        return result
