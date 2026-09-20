import re
from langdetect import detect
import html
import unicodedata
from langdetect.lang_detect_exception import LangDetectException


# HTML Stripping

def html_stripping(text):
    clean_text = re.sub(re.compile('<.*?>'),"",text)
    clean_text = html.unescape(clean_text) # escape characters ko remove kre ga
    return clean_text


# Unicode Normalization
def unicode_normalization(text):
    clean_text = unicodedata.normalize('NFKC',text)
    clean_text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", clean_text)
    return clean_text

# WhiteSpace Removal
def remove_whitespaces(text):
    clean_text = text.replace("\xa0"," ") # normal space
    clean_text = clean_text.replace("\r\n", "\n").replace("\r", "\n") #line endings -> \n
    clean_text = re.sub(r"[ \t]+", " ", clean_text) #tabs and sapces handling
    clean_text = re.sub(r" ?\n ?", "\n", clean_text)  # trim spaces around newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text) #max one blank line between paragraphs
    return clean_text.strip()  

# Language Filtering
def lang_filter(text):
    try:
        if detect(text[:1000]) == 'en':# phle 1000 characters ko check krna kafi hai 
            return text
    except LangDetectException:
        pass
    
    return ""


# Clean Text Pipeline
def clean_text_pipeline(text):
    if not isinstance(text,str):
        return ""
    text = html_stripping(text)
    text = unicode_normalization(text)
    text = remove_whitespaces(text)
    text = lang_filter(text)
    return text
    
    
    
    
    