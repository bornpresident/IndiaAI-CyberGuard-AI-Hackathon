import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = ' '.join(text.split())
        return text
        
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)