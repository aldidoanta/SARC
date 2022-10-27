import nltk
import numpy as np
nltk.download('vader_lexicon')

from eval import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex
from sklearn.linear_model import LogisticRegressionCV as LogitCV
from sklearn import metrics, naive_bayes
from text_embedding.features import *
from text_embedding.vectors import *
from utils import *

def main():
    args = parse()
    if args.dataset.lower() == 'pol':
        SARC = SARC_POL
    elif args.dataset.lower() == 'main':
        SARC = SARC_MAIN

    train_file = SARC+'train-balanced.csv'
    test_file = SARC+'test-balanced.csv'
    comment_file = SARC+'comments.json'

    # Load SARC pol/main sequences with labels.
    print('Load SARC data')
    train_seqs, test_seqs, train_labels, test_labels = \
        load_sarc_responses(train_file, test_file, comment_file, lower=False)
    
    # Ancestor/prior statements that form the context of the sarcasm statements
    train_ancestor = train_seqs['ancestors']
    test_ancestor = test_seqs['ancestors']
    
    # Responses of the ancestor statements
    train_resp = train_seqs['responses']
    test_resp = test_seqs['responses']

    # Split into first and second responses and their labels.
    # {0: list_of_first_responses, 1: list_of_second_responses}
    train_docs = {i: [l[i] for l in train_resp] for i in range(2)}
    test_docs = {i: [l[i] for l in test_resp] for i in range(2)}
    # Convert label values, from {0,1} to {-1,1}
    train_labels = {i: [2*int(l[i])-1 for l in train_labels] for i in range(2)}
    test_labels = {i: [2*int(l[i])-1 for l in test_labels] for i in range(2)}

    # Combine all labels into one array, both in train and test data
    train_all_labels = np.array(train_labels[0] + train_labels[1])
    test_all_labels = np.array(test_labels[0] + test_labels[1])

    # Feature extraction using sentiment analysis (VADER) and emotional affect (NRCLex)
    train_all_docs_sentiment = []
    test_all_docs_sentiment = []
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # Feature extraction for train and test data
    train_all_docs_sentiment = get_extracted_features(train_ancestor, train_docs)
    test_all_docs_sentiment = get_extracted_features(test_ancestor, test_docs)

    # Evaluate this classifier on all responses.
    print('Evaluate the classifier on all responses')
    clf = LogitCV(Cs=[10**i for i in range(-2, 3)], fit_intercept=False, cv=2, dual=np.less(*train_all_docs_sentiment.shape), solver='liblinear', n_jobs=-1, random_state=0)
    clf.fit(train_all_docs_sentiment, train_all_labels)
    print('\tTrain acc: ', clf.score(train_all_docs_sentiment, train_all_labels))
    print('\tTest acc: ', clf.score(test_all_docs_sentiment, test_all_labels))

    # Get vectors for first and second responses.
    n_tr = int(train_all_docs_sentiment.shape[0]/2)
    n_te = int(test_all_docs_sentiment.shape[0]/2)
    train_vecs = {i: train_all_docs_sentiment[i*n_tr:(i+1)*n_tr,:] for i in range(2)}
    test_vecs = {i: test_all_docs_sentiment[i*n_te:(i+1)*n_te,:] for i in range(2)}

    # Final evaluation.
    print('Evaluate the classifier on the original dataset')
    hyperplane = clf.coef_[0,:]
    train_pred_labels = 2*(train_vecs[0].dot(hyperplane) > train_vecs[1].dot(hyperplane))-1
    test_pred_labels = 2*(test_vecs[0].dot(hyperplane) > test_vecs[1].dot(hyperplane))-1
    train_expect_labels = train_labels[0]
    test_expect_labels = test_labels[0]
    print('\tTrain acc: ', (train_pred_labels == train_expect_labels).sum() / train_pred_labels.shape[0])
    print('\tTest acc: ', (test_pred_labels == test_expect_labels).sum() / test_pred_labels.shape[0])

    # Evaluate classifier using NaiveBayes
    gnb = naive_bayes.GaussianNB()
    gnb.fit(train_all_docs_sentiment, train_all_labels)
    y_predict = gnb.predict(test_all_docs_sentiment)
    print('GaussianNB accuracy: ', metrics.accuracy_score(test_all_labels, y_predict))
    print('GaussianNB f-1 score: ', metrics.f1_score(test_all_labels, y_predict))

def get_extracted_features(ancestors, response_docs):
    result = []
    sentiment_analyzer = SentimentIntensityAnalyzer()
    for idx, sentence in enumerate(ancestors):
        previous_statement = sentence[len(sentence) - 1]
        first_response = response_docs[0][idx]
        second_response = response_docs[1][idx]

        # Calculate sentiment scores
        score_previous_statement = sentiment_analyzer.polarity_scores(previous_statement)
        score_first_response = sentiment_analyzer.polarity_scores(first_response)
        score_second_response = sentiment_analyzer.polarity_scores(second_response)

        # Calculate emotion scores
        emotion_score_previous_statement = get_emotion_score(get_preprocessed_sentence(previous_statement))
        emotion_score_first_response = get_emotion_score(get_preprocessed_sentence(first_response))
        emotion_score_second_response = get_emotion_score(get_preprocessed_sentence(second_response))

        # Treat the sentiment results (values of neg, neu, pos) as learning features
        result.append([
            score_previous_statement['neg'],
            score_previous_statement['neu'],
            score_previous_statement['pos'],

            score_first_response['neg'],
            score_first_response['neu'],
            score_first_response['pos'],

            emotion_score_previous_statement['fear'],
            emotion_score_previous_statement['anger'],
            emotion_score_previous_statement['anticipation'],
            emotion_score_previous_statement['trust'],
            emotion_score_previous_statement['surprise'],
            emotion_score_previous_statement['positive'],
            emotion_score_previous_statement['negative'],
            emotion_score_previous_statement['sadness'],
            emotion_score_previous_statement['disgust'],
            emotion_score_previous_statement['joy'],

            emotion_score_first_response['fear'],
            emotion_score_first_response['anger'],
            emotion_score_first_response['anticipation'],
            emotion_score_first_response['trust'],
            emotion_score_first_response['surprise'],
            emotion_score_first_response['positive'],
            emotion_score_first_response['negative'],
            emotion_score_first_response['sadness'],
            emotion_score_first_response['disgust'],
            emotion_score_first_response['joy']
        ])
        result.append([
            score_previous_statement['neg'],
            score_previous_statement['neu'],
            score_previous_statement['pos'],

            score_second_response['neg'],
            score_second_response['neu'],
            score_second_response['pos'],

            emotion_score_previous_statement['fear'],
            emotion_score_previous_statement['anger'],
            emotion_score_previous_statement['anticipation'],
            emotion_score_previous_statement['trust'],
            emotion_score_previous_statement['surprise'],
            emotion_score_previous_statement['positive'],
            emotion_score_previous_statement['negative'],
            emotion_score_previous_statement['sadness'],
            emotion_score_previous_statement['disgust'],
            emotion_score_previous_statement['joy'],

            emotion_score_second_response['fear'],
            emotion_score_second_response['anger'],
            emotion_score_second_response['anticipation'],
            emotion_score_second_response['trust'],
            emotion_score_second_response['surprise'],
            emotion_score_second_response['positive'],
            emotion_score_second_response['negative'],
            emotion_score_second_response['sadness'],
            emotion_score_second_response['disgust'],
            emotion_score_second_response['joy']
        ])
    return np.array(result)

def get_preprocessed_sentence(sentence):
    return lemmatize(list(split_on_punctuation(sentence)))

def get_emotion_score(sentence):
    n_word = len(sentence)
    emo_count = {'fear': 0, 'anger': 0, 'anticipation': 0, 'trust': 0, 'surprise': 0, 'positive': 0, 'negative': 0, 'sadness': 0, 'disgust': 0, 'joy': 0}
    for word in sentence:
        emotion = NRCLex(word).raw_emotion_scores
        for emo in emotion:
            emo_count[emo] += 1
    return {
        'fear': emo_count['fear']/n_word,
        'anger': emo_count['anger']/n_word,
        'anticipation': emo_count['anticipation']/n_word,
        'trust': emo_count['trust']/n_word,
        'surprise': emo_count['surprise']/n_word,
        'positive': emo_count['positive']/n_word,
        'negative': emo_count['negative']/n_word,
        'sadness': emo_count['sadness']/n_word,
        'disgust': emo_count['disgust']/n_word,
        'joy': emo_count['joy']/n_word
    }

main()
