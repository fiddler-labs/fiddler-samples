import pickle

import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups


def train_and_save_model():
    outfile = open('random_forest_model.pickle', 'wb')
    vectorizer_outfile = open('random_forest_vectorizer.pickle', 'wb')

    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        lowercase=False)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, newsgroups_train.target)

    pred = rf.predict(test_vectors)
    score = sklearn.metrics.f1_score(newsgroups_test.target, pred,
                                     average='binary')
    print("Done learning {}".format(score))

    pickle.dump(rf, outfile)
    outfile.close()

    pickle.dump(vectorizer, vectorizer_outfile)
    vectorizer_outfile.close()
