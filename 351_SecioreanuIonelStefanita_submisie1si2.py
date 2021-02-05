import nltk
import pandas as pd
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict


def tokenize(text):
    '''Tokenizator folosit pentru a formata textul
    '''
    return nltk.TweetTokenizer().tokenize(text)


def get_corpus_vocabulary(corpus):
    ''' Returneaza un counter cu toate cuvintele tokenizate din corpusul
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def get_representation(toate_cuvintele, how_many):
    '''
    Functie ce returneaza 2 dictionare cu primele how_many cele mai intalnite cuvinte

    wd2idx     @  che  .   ,   di  e
    idx2wd     0   1   2   3   4   5
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):
    '''Converteste un text intr-un bag of words in care se numara frecventa cuvintelor uzuale in text.
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    '''Functie ce construieste un bow pe baza unui coprus (text de texte)
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)
    return all_features


def write_prediction(out_file, predictions):
    '''Functie ajutatoare ce creaza/scrie intr-un fisier
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed


def split(data, labels, procentaj_valid=0.25):
    '''
    Functie ce face split pe data in proporite de 75% date de antrenare  25% date de testare
    '''
    split_shuffle = ShuffleSplit(n_splits=1, test_size=procentaj_valid, random_state=0)
    train = data
    valid = data

    for i, j in split_shuffle.split(data, labels):
        train = data.loc[i]
        valid = data.loc[j]

    return train, valid


def cross_validate(k, data, labels, estimator):
    '''
        Functie ce face K-Cross-Validation si retrneaza scor f1 (medie armonica intre precizie si recall) pentru fiecare bucata K
        '''
    toate_cuvintele = get_corpus_vocabulary(data)
    wd2idx, idx2wd = get_representation(toate_cuvintele, 1868)

    data1 = corpus_to_bow(data, wd2idx)
    f1 = cross_val_score(estimator, data1, labels, scoring='f1', cv=k)

    return f1


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

'''de sters

toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)
data = corpus_to_bow(corpus, wd2idx)


test_data = corpus_to_bow(test_df['text'], wd2idx)
print(test_data.shape)
'''

labels = train_df['label'].values

multime_de_antrenare_si_validare, multime_de_testare = split(train_df, train_df['label'])
'''Metoda 1 KNN 100 de cuvinte cu K=12 
    schimbati randurile 107 si 139 cu valoarea de 100 in loc de 1868
    si comentati de la 154 pana la 159

'''
# estimator = KNeighborsClassifier(12)
toate_cuvintele = get_corpus_vocabulary(multime_de_antrenare_si_validare['text'])
wd2idx, idx2wd = get_representation(toate_cuvintele, 1868)

data = corpus_to_bow(multime_de_antrenare_si_validare['text'], wd2idx)
# estimator.fit(data, multime_de_antrenare_si_validare['label'])

test_data = corpus_to_bow(multime_de_testare['text'], wd2idx)

# estimator.predict(test_data)

# print(cross_validate(10, multime_de_antrenare_si_validare['text'], multime_de_antrenare_si_validare['label'], estimator))
'''Metoda 2 Naive Bayes Bernoulli cu 1868 de cuvinte 
    schimbati randurile 107 si 139 cu valoarea de 1868 in loc de 100
    pentru a modifica comentati randurile 138 143  147  149  si 153,154'''
# test_data = corpus_to_bow(test_df['text'], wd2idx)
# predictii = estimator.predict(test_data)

estimator = BernoulliNB()
estimator.fit(data, multime_de_antrenare_si_validare['label'])
predictii = estimator.predict(test_data)
print(cross_validate(10, multime_de_antrenare_si_validare['text'], multime_de_antrenare_si_validare['label'], estimator))

write_prediction('submisieProiectIA.csv', predictii)


def cross_validation_predict(k, data, labels, estimator):
    predictii_cross = cross_val_predict(estimator, data, labels, cv=k)

    return predictii_cross


import nltk
import pandas as pd
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict


def tokenize(text):
    '''Tokenizator folosit pentru a formata textul
    '''
    return nltk.TweetTokenizer().tokenize(text)


def get_corpus_vocabulary(corpus):
    ''' Returneaza un counter cu toate cuvintele tokenizate din corpusul
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def get_representation(toate_cuvintele, how_many):
    '''
    Functie ce returneaza 2 dictionare cu primele how_many cele mai intalnite cuvinte

    wd2idx     @  che  .   ,   di  e
    idx2wd     0   1   2   3   4   5
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):
    '''Converteste un text intr-un bag of words in care se numara frecventa cuvintelor uzuale in text.
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    '''Functie ce construieste un bow pe baza unui coprus (text de texte)
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)
    return all_features


def write_prediction(out_file, predictions):
    '''Functie ajutatoare ce creaza/scrie intr-un fisier
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed


def split(data, labels, procentaj_valid=0.25):
    '''
    Functie ce face split pe data in proporite de 75% date de antrenare  25% date de testare
    '''
    split_shuffle = ShuffleSplit(n_splits=1, test_size=procentaj_valid, random_state=0)
    train = data
    valid = data

    for i, j in split_shuffle.split(data, labels):
        train = data.loc[i]
        valid = data.loc[j]

    return train, valid


def cross_validate(k, data, labels, estimator):
    '''
        Functie ce face K-Cross-Validation si retrneaza scor f1 (medie armonica intre precizie si recall) pentru fiecare bucata K
        '''
    toate_cuvintele = get_corpus_vocabulary(data)
    wd2idx, idx2wd = get_representation(toate_cuvintele, 1868)

    data1 = corpus_to_bow(data, wd2idx)
    f1 = cross_val_score(estimator, data1, labels, scoring='f1', cv=k)

    return f1


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

'''de sters

toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)
data = corpus_to_bow(corpus, wd2idx)


test_data = corpus_to_bow(test_df['text'], wd2idx)
print(test_data.shape)
'''

labels = train_df['label'].values

multime_de_antrenare_si_validare, multime_de_testare = split(train_df, train_df['label'])
'''Metoda 1 KNN 100 de cuvinte cu K=12 
    schimbati randurile 107 si 139 cu valoarea de 100 in loc de 1868
    si comentati de la 154 pana la 159

'''
# estimator = KNeighborsClassifier(12)
toate_cuvintele = get_corpus_vocabulary(multime_de_antrenare_si_validare['text'])
wd2idx, idx2wd = get_representation(toate_cuvintele, 1868)

data = corpus_to_bow(multime_de_antrenare_si_validare['text'], wd2idx)
# estimator.fit(data, multime_de_antrenare_si_validare['label'])

test_data = corpus_to_bow(multime_de_testare['text'], wd2idx)

# estimator.predict(test_data)

# print(cross_validate(10, multime_de_antrenare_si_validare['text'], multime_de_antrenare_si_validare['label'], estimator))
'''Metoda 2 Naive Bayes Bernoulli cu 1868 de cuvinte 
    schimbati randurile 107 si 139 cu valoarea de 1868 in loc de 100
    pentru a modifica comentati randurile 138 143  147  149  si 153,154'''
# test_data = corpus_to_bow(test_df['text'], wd2idx)
# predictii = estimator.predict(test_data)

estimator = BernoulliNB()
estimator.fit(data, multime_de_antrenare_si_validare['label'])
predictii = estimator.predict(test_data)
print(cross_validate(10, multime_de_antrenare_si_validare['text'], multime_de_antrenare_si_validare['label'], estimator))

write_prediction('submisieProiectIA.csv', predictii)


def cross_validation_predict(k, data, labels, estimator):
    predictii_cross = cross_val_predict(estimator, data, labels, cv=k)

    return predictii_cross


def matrice_de_confuzie(multime_tweeturi, y_pred):
    adevar_pozitiv = 0
    fals_negativ = 0
    fals_pozitiv = 0
    adevar_negativ = 0

    y_label = multime_tweeturi['label'].values

    for i in range(len(y_label)):
        if (y_pred[i] == 1) and (y_label[i] == 1):
            adevar_pozitiv += 1
        if (y_pred[i] == 0) and (y_label[i] == 1):
            fals_negativ += 1
        if (y_pred[i] == 1) and (y_label[i] == 0):
            fals_pozitiv += 1
        if (y_pred[i] == 0) and (y_label[i] == 0):
            adevar_negativ += 1

    matrice = [[adevar_pozitiv, fals_negativ], [fals_pozitiv, adevar_negativ]]

    print(matrice)


predictii_cross = cross_validation_predict(10, data, multime_de_antrenare_si_validare['label'], estimator)
matrice_de_confuzie(multime_de_antrenare_si_validare, predictii_cross)

#
# [0.84044234 0.83386581 0.82012195 0.8136646  0.84493671 0.81977671] -knn 1868
# [0.87392055 0.85165794 0.85714286 0.86622074 0.88135593 0.83277592] -bernoulli 1868

# [0.87223169 0.85304659 0.8441331  0.85472973 0.87889273 0.85328836] -Mn 6000 ambele

# [0.87223169 0.85304659 0.8441331  0.85472973 0.87889273 0.85328836]-Mn 6000,5000

# [0.88041594 0.86120996 0.84429066 0.85666105 0.87737478 0.85762144]-Mn 5000 ambele
# [0.8777969  0.86619718 0.84283247 0.85234899 0.87586207 0.85475793]- Mn 4500
# [0.87457627 0.85714286 0.84507042 0.85714286 0.87847222 0.8537415 ] Mn 7000 ambele
# [0.87628866 0.85314685 0.86254296 0.85025818 0.87279152 0.84320557] MN 10000 ambele
# [0.8815331  0.85361552 0.85961872 0.86294416 0.86678508 0.84561404] Mn 11000

# [0.87628866 0.84751773 0.85172414 0.85761589 0.87521368 0.85618729]


# [0.84310618 0.82408875 0.82043344 0.81259843 0.85759494 0.82747604]
# [0.81849315 0.81109185 0.83082077 0.80872483 0.84577114 0.80730897]

# [0.84310618 0.82408875 0.82043344 0.81259843 0.85759494 0.82747604]
# [0.81849315 0.81109185 0.83082077 0.80872483 0.84577114 0.80730897]
#



predictii_cross = cross_validation_predict(10, data, multime_de_antrenare_si_validare['label'], estimator)
matrice_de_confuzie(multime_de_antrenare_si_validare, predictii_cross)

#
# [0.84044234 0.83386581 0.82012195 0.8136646  0.84493671 0.81977671] -knn 1868
# [0.87392055 0.85165794 0.85714286 0.86622074 0.88135593 0.83277592] -bernoulli 1868

# [0.87223169 0.85304659 0.8441331  0.85472973 0.87889273 0.85328836] -Mn 6000 ambele

# [0.87223169 0.85304659 0.8441331  0.85472973 0.87889273 0.85328836]-Mn 6000,5000

# [0.88041594 0.86120996 0.84429066 0.85666105 0.87737478 0.85762144]-Mn 5000 ambele
# [0.8777969  0.86619718 0.84283247 0.85234899 0.87586207 0.85475793]- Mn 4500
# [0.87457627 0.85714286 0.84507042 0.85714286 0.87847222 0.8537415 ] Mn 7000 ambele
# [0.87628866 0.85314685 0.86254296 0.85025818 0.87279152 0.84320557] MN 10000 ambele
# [0.8815331  0.85361552 0.85961872 0.86294416 0.86678508 0.84561404] Mn 11000

# [0.87628866 0.84751773 0.85172414 0.85761589 0.87521368 0.85618729]


# [0.84310618 0.82408875 0.82043344 0.81259843 0.85759494 0.82747604]
# [0.81849315 0.81109185 0.83082077 0.80872483 0.84577114 0.80730897]

# [0.84310618 0.82408875 0.82043344 0.81259843 0.85759494 0.82747604]
# [0.81849315 0.81109185 0.83082077 0.80872483 0.84577114 0.80730897]
#
