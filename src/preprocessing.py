import re

# HTML Stripping

def html_stripping(text):
    clean_text = re.sub(re.compile('<.*?>'),text)
    return clean_text


# Unicode Normalization
def unicode_normalization(text):
    clean_text = text.encode('utf_8')
    return clean_text