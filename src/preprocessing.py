import re

# HTML Stripping

def html_stripping(text):
    clean_text = re.sub(re.compile('<.*?>'),text)
    return clean_text
    