import pandas as pd
import string
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

#
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')


# my_analyzer replaces both the preprocessor and tokenizer
# it also replaces stop word removal and ngram constructions

def my_analyzer(s):
    # Synonym List
    syns = {'veh': 'vehicle', 'car': 'vehicle', 'chev': 'cheverolet', \
            'chevy': 'cheverolet', 'air bag': 'airbag', \
            'seat belt': 'seatbelt', "n't": 'not', 'to30': 'to 30', \
            'wont': 'would not', 'cant': 'can not', 'cannot': 'can not', \
            'couldnt': 'could not', 'shouldnt': 'should not', \
            'wouldnt': 'would not', }

    # Preprocess String s
    s = s.lower()
    s = s.replace(',', '. ')
    # Tokenize
    tokens = word_tokenize(s)
    tokens = [word.replace(',', '') for word in tokens]
    tokens = [word for word in tokens if ('*' not in word) and \
              ("''" != word) and ("``" != word) and \
              (word != 'description') and (word != 'dtype') \
              and (word != 'object') and (word != "'s")]

    # Map synonyms
    for i in range(len(tokens)):
        if tokens[i] in syns:
            tokens[i] = syns[tokens[i]]

    # Remove stop words
    punctuation = list(string.punctuation) + ['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    stop = stopwords.words('english') + punctuation + pronouns
    filtered_terms = [word for word in tokens if (word not in stop) and \
                      (len(word) > 1) and (not word.replace('.', '', 1).isnumeric()) \
                      and (not word.replace("'", '', 2).isnumeric())]

    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N': wn.NOUN, 'J': wn.ADJ, 'V': wn.VERB, 'R': wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos = tagged_token[1]
        pos = pos[0]
        try:
            pos = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    return stemmed_tokens


# Further Customization of Stopping and Stemming using NLTK
def my_preprocessor(s):
    # Vectorizer sends one string at a time
    s = s.lower()
    s = s.replace(',', '. ')
    print("preprocessor")
    return (s)


def my_tokenizer(s):
    # Tokenize
    print("Tokenizer")
    tokens = word_tokenize(s)
    tokens = [word.replace(',', '') for word in tokens]
    tokens = [word for word in tokens if word.find('*') != True and \
              word != "''" and word != "``" and word != 'description' \
              and word != 'dtype']
    return tokens


# Increase Pandas column width to let pandas read large text columns
pd.set_option('max_colwidth', 32000)
# Read GMC Ignition Recall Comments from NTHSA Data
#file_path = '/Users/Home/Desktop/python/Excel/'
df = pd.read_excel("wine.xlsx")

# Setup simple constants
n_docs     = len(df['description'])
n_samples  = n_docs
m_features = None
s_words    = 'english'
ngram = (1,2)

# Setup reviews in list 'discussions'
discussions = []
for i in range(n_samples):
    discussions.append(("%s" %df['description'].iloc[i]))

# Create Word Frequency by Review Matrix using Custom Analyzer
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=m_features,\
                     analyzer=my_analyzer, ngram_range=ngram)
tf = cv.fit_transform(discussions)

print("\nVectorizer Parameters\n", cv, "\n")


# LDA For Term Frequency x Doc Matrix
n_topics        = 15
max_iter        =  5
learning_offset = 20.
learning_method = 'online'
# In sklearn, LDA is synonymous with SVD (according to their doc)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,\
                                learning_method=learning_method, \
                                learning_offset=learning_offset, \
                                random_state=12345)
lda.fit_transform(tf)
print('{:.<22s}{:>6d}'.format("Number of Reviews", tf.shape[0]))
print('{:.<22s}{:>6d}'.format("Number of Terms",     tf.shape[1]))
print("\nTopics Identified using LDA")
tf_features = cv.get_feature_names()
max_words = 15
for topic_idx, topic in enumerate(lda.components_):
        message  = "Topic #%d: " %topic_idx
        message += " ".join([tf_features[i]
                             for i in topic.argsort()[:-max_words - 1:-1]])
        print(message)
        print()

# LDA for TF-IDF x Doc Matrix
# First Create Term-Frequency/Inverse Doc Frequency by Review Matrix
# This requires constructing Term Freq. x Doc. matrix first
tf_idf = TfidfTransformer()
print("\nTF-IDF Parameters\n", tf_idf.get_params(),"\n")
tf_idf = tf_idf.fit_transform(tf)
# Or you can construct the TF/IDF matrix from the data
tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=2, max_features=m_features,\
                             analyzer=my_analyzer, ngram_range=ngram)
tf_idf = tfidf_vect.fit_transform(discussions)
print("\nTF_IDF Vectorizer Parameters\n", tfidf_vect, "\n")

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,\
                                learning_method=learning_method, \
                                learning_offset=learning_offset, \
                                random_state=12345)
lda.fit_transform(tf_idf)
print('{:.<22s}{:>6d}'.format("Number of Reviews", tf.shape[0]))
print('{:.<22s}{:>6d}'.format("Number of Terms",     tf.shape[1]))
print("\nTopics Identified using LDA with TF_IDF")
tf_features = cv.get_feature_names()
max_words = 15
for topic_idx, topic in enumerate(lda.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([tf_features[i]
                             for i in topic.argsort()[:-max_words - 1:-1]])
        print(message)
        print()


