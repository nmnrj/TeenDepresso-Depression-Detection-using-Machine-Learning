import re
import string
import contractions
import emoji
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")
stop_words.discard("never")

lemmatizer = WordNetLemmatizer()


def lowercase_text(text):
    return text.lower()


def expand_contractions(text):
    return contractions.fix(text)


def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text)


def remove_html(text):
    return re.sub(r"<.*?>", "", text)


def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")


def normalize_repeated_characters(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN


def lemmatize_words(tokens):

    tagged_tokens = pos_tag(tokens)

    lemmatized = []

    for word, tag in tagged_tokens:

        pos = get_wordnet_pos(tag)

        lemma = lemmatizer.lemmatize(word, pos)

        lemmatized.append(lemma)

    return lemmatized

def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = lowercase_text(text)

    text = expand_contractions(text)

    text = remove_urls(text)

    text = remove_html(text)

    text = remove_emojis(text)

    text = normalize_repeated_characters(text)

    text = remove_punctuation(text)

    text = remove_numbers(text)

    text = normalize_whitespace(text)

    if text == "":
        return ""

    tokens = tokenize_text(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize_words(tokens)

    clean_text = " ".join(tokens)

    clean_text = normalize_whitespace(clean_text)

    return clean_text


if __name__ == "__main__":

    samples = [

        "I'm feeling sooo depressed 😢😢 Visit https://google.com <p>I haven't been sleeping for 2 days.</p>",

        "I don't want to live anymore.",

        "Nobody understands me!!!!",

        "I am NOT happy.",

        "I have never felt this lonely.",

        "Life is beautiful 😊",

        "I can't concentrate on anything.",

        "Everything is getting worse.....",

        "I love Machine Learning ❤️"

    ]

    print("=" * 80)
    print("TEXT PREPROCESSING TEST")
    print("=" * 80)

    for i, sentence in enumerate(samples, start=1):

        print(f"\nExample {i}")
        print("-" * 80)

        print("Original :")
        print(sentence)

        print("\nProcessed :")
        print(preprocess_text(sentence))