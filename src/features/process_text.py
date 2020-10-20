from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
from textblob import TextBlob
import lemmy


def preprocess_text(text):
    # text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = str(text).lower().strip()

    # caveat: this might conflict with the english text
    da_stop_words = stopwords.words('danish')
    stemmer = DanishStemmer()
    lemmatizer = lemmy.load("da")

    # remove plurals
    textblob = TextBlob(text)
    singles = [stemmer.stem(word) for word in textblob.words]

    # remove danish stopwords
    no_stop_words = [word for word in singles if word not in da_stop_words]

    # join text so it can be lemmatized
    joined_text = " ".join(no_stop_words)

    # lemmatization
    final_text = lemmatizer.lemmatize("", joined_text)

    return final_text[0]
