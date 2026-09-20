from src.preprocessing import (
    html_stripping,
    unicode_normalization,
    remove_whitespaces,
    lang_filter,
    clean_text_pipeline,
)




# HTML Stripping testing
def test_html_tags_removed():
    assert html_stripping("<p>Hello <b>world</b></p>").strip() == "Hello world"



# Unicode Normalzation Testing 
def test_unicode_ligature():
    assert unicode_normalization("ﬁne") == "fine"

def test_unicode_zero_width_removed():
    assert unicode_normalization("hel\u200blo") == "hello"


# Whitespace Removal Testing 
def test_extra_spaces_removed():
    assert remove_whitespaces("a     b") == "a b"

def test_extra_blank_lines_removed():
    assert remove_whitespaces("a\n\n\n\nb") == "a\n\nb"

def test_start_end_spaces_removed():
    assert remove_whitespaces("   hello   ") == "hello"


# Filter Language testing 
ENGLISH = "Machine learning helps computers learn patterns from data and improve over time."
FRENCH = "L'apprentissage automatique aide les ordinateurs à apprendre à partir des données."
def test_english_kept():
    assert lang_filter(ENGLISH) == ENGLISH

def test_french_dropped():
    assert lang_filter(FRENCH) == ""

def test_empty_text_no_crash():
    assert lang_filter("") == ""


# Full Preprocessing pipeline testing
def test_pipeline():
    raw = "<p>" + ENGLISH + "     more</p>"
    result = clean_text_pipeline(raw)
    assert "<" not in result
    
    assert "   " not in result