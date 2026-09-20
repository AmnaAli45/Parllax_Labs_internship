import re
from langdetect import detect

# HTML Stripping

def html_stripping(text):
    clean_text = re.sub(re.compile('<.*?>'),text)
    return clean_text


# Unicode Normalization
def unicode_normalization(text):
    clean_text = text.encode('utf_8')
    return clean_text

# WhiteSpace Removal
def remove_whitespaces(text):
    clean_text = text.replace("\xa0"," ") # normal space
    clean_text = clean_text.replace("\r\n","\n") #line endings -> \n
    clean_text = re.sub(r"[\t]+", " ", clean_text) #tabs handling
    clean_text = re.sub(r" ?\n ?", "\n", clean_text)  # trim spaces around newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text) #max one blank line between paragraphs
    return clean_text.strip()  

# Language Filtering
def lang_filter(text):
    if detect(text[:1000]) == 'en':# phle 1000 characters ko check krna kafi hai 
        return text
    else: 
        return ""


    