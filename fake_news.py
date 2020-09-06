import pickle
import re

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def clean_dataset(df_data):
    df_data.loc[df_data['label'] == 'Fake', ['label']] = 'FAKE'
    df_data.loc[df_data['label'] == 'fake', ['label']] = 'FAKE'
    df_data.loc[df_data['source'] == 'facebook', ['source']] = 'Facebook'
    df_data.text.fillna(df_data.title, inplace=True)

    df_data.loc[5]['label'] = 'FAKE'
    df_data.loc[15]['label'] = 'TRUE'
    df_data.loc[43]['label'] = 'FAKE'
    df_data.loc[131]['label'] = 'TRUE'
    df_data.loc[242]['label'] = 'FAKE'

    df_data = df_data.sample(frac=1).reset_index(drop=True)
    df_data.title.fillna('missing', inplace=True)
    df_data.source.fillna('missing', inplace=True)

    df_data['title_text'] = df_data['title'] + ' ' + df_data['text']
    df['title_text'] = df['title_text'].apply(preprocessor)
    return df_data


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


if __name__ == '__main__':
    df = clean_dataset(pd.read_csv('corona_fake.csv'))
    tfidf = TfidfVectorizer(
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=tokenizer_porter,
        use_idf=True,
        norm='l2',
        smooth_idf=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf.fit_transform(df['title_text']),
        df.label.values,
        random_state=0,
        test_size=0.5,
        shuffle=False,
    )

    clf = LogisticRegressionCV(
        cv=5,
        scoring='accuracy',
        random_state=0,
        n_jobs=-1,
        verbose=3,
        max_iter=300,
    ).fit(X_train, y_train)

    fake_news_model = open('fake_news_model.sav', 'wb')
    pickle.dump(clf, fake_news_model)
    fake_news_model.close()

    saved_clf = pickle.load(open('fake_news_model.sav', 'rb'))

    print(saved_clf.score(X_test, y_test))

    # test model
    y_pred = clf.predict(X_test)
    print('---Test Set Results---')
    print('Accuracy with logreg: {}'.format(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
