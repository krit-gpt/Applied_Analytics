# Data
# To illustrate the basic concepts behind the analysis of text, a short collection of 8 documents will be used available from http://www.gutenberg.org .

# Import Packages
import string
import re
import nltk
# NTLK is the Natural Language Tool Kit
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
# The following two packages are used to sort the term/doc matrix
from collections import Counter
import operator
# Download NLTK Supporting Files
# The NLTK package uses several supporting files. These need to be downloaded, but only once. Download them initially using the following statements. After these execute successfully, comment them out of your code.

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
# [nltk_data] Downloading package punkt to /Users/Home/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# [nltk_data] Downloading package averaged_perceptron_tagger to
# [nltk_data]     /Users/Home/nltk_data...
# [nltk_data]   Package averaged_perceptron_tagger is already up-to-
# [nltk_data]       date!
# [nltk_data] Downloading package stopwords to /Users/Home/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# [nltk_data] Downloading package wordnet to /Users/Home/nltk_data...
# [nltk_data]   Package wordnet is already up-to-date!
True
Create Program Control Attributes
The files list is a list of the documents that will be processed. The remaing attributes are used to turn on and off tagging, stop words and stemming.

file_path = '/Users/Home/Desktop/python/Text/TextFiles/'
files = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', \
         'T7.txt', 'T8.txt']
pos_tags = True
stemming = True
remove_stop = True
Tokenization, POS Tagging, Stop Removal & Stemming
With 8 documents, it is best to do everything inside a large loop, reading each document and then processing that document before creating the final term/document matrix.

term_doc = []
for file in files:
    with open (file_path+file, "r") as text_file:
        adoc = text_file.read()
    # Convert to all lower case - required
    adoc = ("%s" %adoc).lower()
    # Replace special characters with spaces
    adoc = adoc.replace('-', ' ')
    adoc = adoc.replace('_', ' ')
    adoc = adoc.replace(',', ' ')
    # Replace not contraction with not
    adoc = adoc.replace("'nt", " not")
    adoc = adoc.replace("n't", " not")
    # Tokenize
    tokens = word_tokenize(adoc)
    tokens = [word.replace(',', '') for word in tokens]
    tokens = [word for word in tokens if ('*' not in word) and \
              word != "''" and word !="``"]
    for word in tokens:
        word = re.sub(r'[^\w\d\s]+','',word)
    print("\nDocument "+file+" contains a total of", len(tokens), " terms.")
    
    if pos_tags:
        # POS Tagging
        tokens = nltk.pos_tag(tokens)
    if remove_stop:
        # Remove stop words
        stop = stopwords.words('english') + list(string.punctuation)
        stop.append("said")
        tokens = [word for word in tokens if word[0] not in stop]
        # Remove single character words and simple punctuation
        tokens = [word for word in tokens if len(word) > 1]
        # Remove numbers and possive "'s"
        tokens = [word for word in tokens \
                       if (not word[0].replace('.','',1).isnumeric()) and \
                       word[0]!="'s" ]
    if stemming:
        # Lemmatization - Stemming with POS
        # WordNet Lematization Stems using POS
        stemmer = SnowballStemmer("english")
        wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
        wnl = WordNetLemmatizer()
        stemmed_tokens = []
        for token in tokens:
            term = token[0]
            pos  = token[1]
            pos  = pos[0]
            try:
                pos   = wn_tags[pos]
                stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
            except:
                stemmed_tokens.append(stemmer.stem(term))
    if stemming:
        print("Document "+file+" contains", len(stemmed_tokens), \
                  "terms after stemming.") 
        tokens = stemmed_tokens
    # Word distribution
    #fdist = FreqDist(word for word in stemmed_tokens)
    fdist = FreqDist(tokens)
    # Use with Wordnet
    td= {}
    for word, freq in fdist.most_common(2000):
        td[word] = freq
    term_doc.append(td)
Document T1.txt contains a total of 86512  terms.
Document T1.txt contains 40072 terms after stemming.

Document T2.txt contains a total of 108475  terms.
Document T2.txt contains 48290 terms after stemming.

Document T3.txt contains a total of 104764  terms.
Document T3.txt contains 50163 terms after stemming.

Document T4.txt contains a total of 83138  terms.
Document T4.txt contains 35273 terms after stemming.

Document T5.txt contains a total of 76236  terms.
Document T5.txt contains 35031 terms after stemming.

Document T6.txt contains a total of 35074  terms.
Document T6.txt contains 15740 terms after stemming.

Document T7.txt contains a total of 80268  terms.
Document T7.txt contains 35592 terms after stemming.

Document T8.txt contains a total of 64518  terms.
Document T8.txt contains 29804 terms after stemming.
Create the Term/Document Matrix
The following code creates the term/document matrix by combining the counts for each document. A list of dictionaries term_doc was created for the 8 documents. Each dictionary contains the terms for that document and the count of the number of times it appears in that document.

td_mat = {}
for td in term_doc:
    td_mat = Counter(td_mat)+Counter(td)
td_matrix = {}
for k, v in td_mat.items():
    td_matrix[k] = [v]
for td in term_doc:
    for k, v in td_matrix.items():
        if k in td:
            td_matrix[k].append(td[k])
        else:
            td_matrix[k].append(0)
Sort Term/Document Matrix by Total Word Count
The term/doc matrix td_matrix must be sorted placing the terms with the largest total word count at the top of the matrix since we want to display the top 20 terms. The 20 terms with the largest word counts.

Sorting is done using operator.itemgetter() which sorts a dictionary and returns the result as a list. td_matrix_sorted is a list, not a dictionary.

td_matrix_sorted = sorted(td_matrix.items(), key=operator.itemgetter(1),\
                          reverse=True)
Display the Top 20 Terms
print("Scenario: POS=", pos_tags, "Remove Stop Words=", remove_stop, \
      " Stemming=", stemming)
print("------------------------------------------------------------")
print("     TERM      TOTAL   T1   T2   T3   T4   T5   T6   T7   T8")
for i in range(20):
    s = '{:<15s}'.format(td_matrix_sorted[i][0])
    v = td_matrix_sorted[i][1]
    for j in range(9):
        s = s + '{:>5d}'.format(v[j])
    print('{:<60s}'.format(s))
print("____________________________________________________________")
Scenario: POS= True Remove Stop Words= True  Stemming= True
------------------------------------------------------------
     TERM      TOTAL   T1   T2   T3   T4   T5   T6   T7   T8
one             2127  291  437  348  211  312  121  202  205
water           2040   47  922  825    7   94    7   55   83
make            1928  204  694  262  185  237   63  169  114
would           1855  270  407  195  309  222   60  289  103
go              1620  212  292   18  239  154  103  374  228
come            1511  211  153   62  126  276  155  282  246
could           1363  221  121   49  364  195   93  203  117
time            1333  137  128  175  167  164  213  216  133
see             1188  179  232  129  156  110   72  172  138
light           1175   87  461  322   21   92   61   60   71
get             1147  171  291   24   76  121   53  316   95
air             1126   69  518  412   20   19   23   30   35
know            1043  165  102  112  223  119   46  203   73
day              939   87   52   82  117  337   49  107  108
take             926  129  174   82  110  135   52  178   66
upon             891  168   11  163   88   28  113  148  172
way              844   78  211  122   73  100   42  116  102
thing            832  120  241   10   76   57  100  113  115
like             828  173  119   56  100   80   74  130   96
man              827  248   38  104   90   90   70   62  125
____________________________________________________________