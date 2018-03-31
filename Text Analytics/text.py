import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
import matplotlib



# #Download nltk supporting packages
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')

#open file

for i in range(1,9):

    print("--------------------------------------------------------")
    print("---------------------- DOCUMENT" + str(i) +  " -----------------------")
    print("--------------------------------------------------------")


    with open ("T" + str(i) + ".txt", "r") as text_file:
        adoc = text_file.read()


    # Convert to all lower case - required
    #a_discussion = ("%s" %df[0:1]).lower()
    a_discussion = ("%s" %adoc).lower()
    a_discussion = a_discussion.replace('-', ' ')
    a_discussion = a_discussion.replace('_', ' ')
    a_discussion = a_discussion.replace(',', ' ')
    a_discussion = a_discussion.replace("'nt", " not")

    # Tokenize
    tokens = word_tokenize(a_discussion)
    tokens = [word.replace(',', '') for word in tokens]
    tokens = [word for word in tokens if ('*' not in word) and word != "''" and \
              word !="``"]
    # Remove punctuation
    for word in tokens:
        word = re.sub(r'[^\w\d\s]+','',word)

    print("\nDocument contains a total of", len(tokens), " terms.")


    # POS Tagging (POS - Parts of Speech)
    # tagged_tokens = nltk.pos_tag(tokens)
    # pos_list = [word[1] for word in tagged_tokens if word[1] != ":" and \
    #             word[1] != "."]
    # pos_dist = FreqDist(pos_list)
    # # pos_dist.plot(title="Parts of Speech")
    # for pos, frequency in pos_dist.most_common(pos_dist.N()):
    #     print('{:<15s}:{:>4d}'.format(pos, frequency))


    # Remove stop words
    stop = stopwords.words('english') + list(string.punctuation)
    # stop_tokens = [word for word in tagged_tokens if word[0] not in stop]
    # Remove single character words and simple punctuation
    stop_tokens = [word for word in tokens if len(word) > 1]
    # Remove numbers and possive "'s"
    stop_tokens = [word for word in stop_tokens \
                   if (not word[0].replace('.','',1).isnumeric()) and \
                   word[0]!="'s" ]
    print("\nDocument contains", len(stop_tokens), \
                          " terms after removing stop words.\n")
    token_dist = FreqDist(stop_tokens)

    for word, frequency in token_dist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word[0], frequency))


    # # Lemmatization - Stemming with POS
    # # WordNet Lematization Stems using POS
    # stemmer = SnowballStemmer("english")
    # wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    # wnl = WordNetLemmatizer()
    # stemmed_tokens = []
    # for token in stop_tokens:
    #     term = token[0]
    #     pos  = token[1]
    #     pos  = pos[0]
    #     try:
    #         pos   = wn_tags[pos]
    #         stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
    #     except:
    #         stemmed_tokens.append(stemmer.stem(term))
    # print("Document contains", len(stemmed_tokens), "terms after stemming.\n")



    # Word distribution
    #fdist = FreqDist(word for word in stemmed_tokens)
    fdist = FreqDist(stop_tokens)
    # Use with Wordnet
    for word, freq in fdist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word, freq))
    # Use with Simple Steming, not with WordNet
    #for word, freq in fdist.most_common(20):
        # print('{:<15s}:{:>4d}'.format(word[0], freq))
    fdist_top = nltk.probability.FreqDist()
    for word, freq in fdist.most_common(20):
        fdist_top[word] = freq
    # fdist_top.plot()







