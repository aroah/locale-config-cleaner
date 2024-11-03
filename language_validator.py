from langdetect import detect, LangDetectException

class LanguageValidator:
    def __init__(self):
        self.language_mapping = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese'
        }

    def detect_language(self, text):
        """Detect the language of given text."""
        try:
            if not isinstance(text, str) or not text.strip():
                return None
            return detect(text)
        except LangDetectException:
            return None

    def validate_dataframe(self, df):
        """Validate language content in the DataFrame."""
        issues = []
        
        # Assume first column is key, others are language specific
        language_columns = df.columns[1:]
        
        for idx, row in df.iterrows():
            for col in language_columns:
                expected_lang = col.split('_')[0].lower()  # Assume column names like 'en_text', 'es_text'
                
                if expected_lang in self.language_mapping:
                    detected_lang = self.detect_language(str(row[col]))
                    
                    if detected_lang and detected_lang != expected_lang:
                        issues.append({
                            'row': idx + 1,
                            'column': col,
                            'message': f"Expected {self.language_mapping[expected_lang]} but detected {self.language_mapping.get(detected_lang, detected_lang)}"
                        })

        return issues
