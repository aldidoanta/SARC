import nltk
nltk.download('stopwords')
from eval import parse, preprocessing
from emotion_extraction import extract_emo, average_emo
from turn_level_sentiment import get_extracted_features
from nltk.corpus import  stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import normalize
from text_embedding.features import *
from text_embedding.vectors import *
from utils import *

def main():
    args = parse()
    if args.dataset.lower() == 'pol':
        SARC = SARC_POL
    elif args.dataset.lower() == 'main':
        SARC = SARC_MAIN

    SARC = SARC_POL
    train_file = SARC+'train-balanced.csv'
    test_file = SARC+'test-balanced.csv'
    comment_file = SARC+'comments.json'

    results = []
    results.append(extract_bong(train_file, test_file, comment_file, args))
    results.append(extract_emotion_features(train_file, test_file, comment_file, args))
    results.append(extract_sentiment(train_file, test_file, comment_file))

    print("Feature\t\t|Accuracy\t\t|Precision\t\t|Recall\t\t\t|F1-Score\t")
    for row in range(len(results)):
        if row == 0:
            model = "BONG\t\t|"
        elif row == 1:
            model = "Emotion\t\t|"
        else:
            model="Sentiment\t|"

        print(model, results[row][0], "\t", results[row][1], "\t", results[row][2], "\t", results[row][3], "\t")

    print("\t\t\tFeature Importance")
    for row in range(len(results)):
        if row == 0:
            model = "BONG\t\t|"
        elif row == 1:
            model = "Emotion\t\t|"
        else:
            model="Sentiment\t|"

        print(model, results[row][4])

def extract_bong(train_file, test_file, comment_file, args):

    # Load SARC pol/main sequences with labels.
    train_seqs, test_seqs, train_labels, test_labels = \
        load_sarc_responses(train_file, test_file, comment_file, lower=args.lower)

    # Only use responses for this method. Ignore ancestors.
    train_resp = train_seqs['responses']
    test_resp = test_seqs['responses']

    # Split into first and second responses and their labels.
    # {0: list_of_first_responses, 1: list_of_second_responses}
    train_docs = {i: [l[i] for l in train_resp] for i in range(2)}
    test_docs = {i: [l[i] for l in test_resp] for i in range(2)}
    train_labels = {i: [2*int(l[i])-1 for l in train_labels] for i in range(2)}
    test_labels = {i: [2*int(l[i])-1 for l in test_labels] for i in range(2)}

    # Train a classifier on all responses in training data. We will later use this
    # classifier to determine for every sequence which of the 2 responses is more sarcastic.
    train_all_docs_tok = preprocessing(tokenize(train_docs[0] + train_docs[1]))
    test_all_docs_tok = preprocessing(tokenize(test_docs[0] + test_docs[1]))
    train_all_labels = np.array(train_labels[0] + train_labels[1])
    test_all_labels = np.array(test_labels[0] + test_labels[1])

    # Bongs or embeddings.
    if args.embed:
        #print('Create embeddings')
        weights = None
        if args.weights == 'sif':
            weights = sif_weights(train_all_docs_tok, 1E-3)
        if args.weights == 'snif':
            weights = sif_weights(train_all_docs_tok, 1E-3)
            weights = {f: 1-w for f, w in weights.items()}
        w2v = vocab2vecs({word for doc in train_all_docs_tok+test_all_docs_tok for word in doc}, vectorfile=args.embedding)
        train_all_vecs = docs2vecs(train_all_docs_tok, f2v=w2v, weights=weights)
        test_all_vecs = docs2vecs(test_all_docs_tok, f2v=w2v, weights=weights)
    else:
        #print('Create bongs')
        n = args.n
        min_count = args.min_count
        train_ngrams = [sum((list(nltk.ngrams(doc, k)) for k in range(1, n+1)), []) for doc in train_all_docs_tok]
        test_ngrams = [sum((list(nltk.ngrams(doc, k)) for k in range(1, n+1)), []) for doc in test_all_docs_tok]
        vocabulary = feature_vocab(train_ngrams, min_count=min_count)
        train_all_vecs = docs2bofs(train_ngrams, vocabulary)
        test_all_vecs = docs2bofs(test_ngrams, vocabulary)

    # Normalize?
    if args.normalize:
        normalize(train_all_vecs, copy=False)
        normalize(test_all_vecs, copy=False)
    #print('Dimension of representation: %d'%train_all_vecs.shape[1])

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_all_vecs, train_all_labels)
    predict = clf.predict(test_all_vecs)
    feature_importance = clf.feature_importances_

    precision = precision_score(test_all_labels, predict)
    recall = recall_score(test_all_labels, predict)
    f1 = f1_score(test_all_labels, predict)
    accuracy = accuracy_score(test_all_labels, predict)

    return accuracy, precision, recall, f1, feature_importance

def extract_emotion_features(train_file, test_file, comment_file, args):
    # Load SARC pol/main sequences with labels.
    train_seqs, test_seqs, train_labels, test_labels = \
        load_sarc_responses(train_file, test_file, comment_file, lower=args.lower)

    # Ancestor/prior statements that form the context of the sarcasm statements
    train_ancestor = train_seqs['ancestors']
    test_ancestor = test_seqs['ancestors']

    # Only use responses for this method. Ignore ancestors.
    train_resp = train_seqs['responses']
    test_resp = test_seqs['responses']

    # Split into first and second responses and their labels.
    # {0: list_of_first_responses, 1: list_of_second_responses}
    train_docs = {i: [l[i] for l in train_resp] for i in range(2)}
    test_docs = {i: [l[i] for l in test_resp] for i in range(2)}
    train_labels = {i: [2*int(l[i])-1 for l in train_labels] for i in range(2)}
    test_labels = {i: [2*int(l[i])-1 for l in test_labels] for i in range(2)}

    train_anc_docs = {0: [l[0] for l in train_ancestor]}
    test_anc_docs = {0: [l[0] for l in test_ancestor]}

    # Train a classifier on all responses in training data. We will later use this
    # classifier to determine for every sequence which of the 2 responses is more sarcastic.
    train_all_ancs_tok = preprocessing(tokenize(train_anc_docs[0]))
    test_all_ancs_tok = preprocessing(tokenize(test_anc_docs[0]))
    train_all_docs_tok = preprocessing(tokenize(train_docs[0] + train_docs[1]))
    test_all_docs_tok = preprocessing(tokenize(test_docs[0] + test_docs[1]))
    train_all_labels = np.array(train_labels[0] + train_labels[1])
    test_all_labels = np.array(test_labels[0] + test_labels[1])

    train_all_docs_emo = []
    test_all_docs_emo = []
    # Measuring Emotions in each sentence
    for idx, sentence in enumerate(train_ancestor):
        previous_statement = train_all_ancs_tok[idx]
        next_statement = preprocessing(tokenize([train_docs[0][idx], train_docs[1][idx]]))
        first_response = next_statement[0]
        second_response = next_statement[1]

        emo_prev = extract_emo(previous_statement)
        emo_first = extract_emo(first_response)
        emo_second = extract_emo(second_response)

        train_all_docs_emo.append(average_emo(emo_prev, emo_first, len(previous_statement), len(first_response)))
        train_all_docs_emo.append(average_emo(emo_prev, emo_second, len(previous_statement), len(second_response)))

    train_all_docs_emo = np.array(train_all_docs_emo)

    for idx, sentence in enumerate(test_ancestor):
        previous_statement = test_all_ancs_tok[idx]
        next_statement = preprocessing(tokenize([test_docs[0][idx], test_docs[1][idx]]))
        first_response = next_statement[0]
        second_response = next_statement[1]

        emo_prev = extract_emo(previous_statement)
        emo_first = extract_emo(first_response)
        emo_second = extract_emo(second_response)

        test_all_docs_emo.append(average_emo(emo_prev, emo_first, len(previous_statement), len(first_response)))
        test_all_docs_emo.append(average_emo(emo_prev, emo_second, len(previous_statement), len(second_response)))

    test_all_docs_emo = np.array(test_all_docs_emo)

    # Evaluate this classifier on all responses.
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_all_docs_emo, train_all_labels)
    predict = clf.predict(test_all_docs_emo)
    feature_importance = clf.feature_importances_

    # Measure Performance
    precision = precision_score(test_all_labels, predict)
    recall = recall_score(test_all_labels, predict)
    f1 = f1_score(test_all_labels, predict)
    accuracy = accuracy_score(test_all_labels, predict)

    return accuracy, precision, recall, f1, feature_importance

def extract_sentiment(train_file, test_file, comment_file):
    # Load SARC pol/main sequences with labels.
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

    # Feature extraction for train and test data, using sentiment analysis (VADER) and emotional affect (NRCLex)
    train_all_docs_sentiment = get_extracted_features(train_ancestor, train_docs)
    test_all_docs_sentiment = get_extracted_features(test_ancestor, test_docs)

    # Evaluate this classifier on all responses.
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_all_docs_sentiment, train_all_labels)
    predict = clf.predict(test_all_docs_sentiment)
    feature_importance = clf.feature_importances_

    # Measure Performance
    precision = precision_score(test_all_labels, predict)
    recall = recall_score(test_all_labels, predict)
    f1 = f1_score(test_all_labels, predict)
    accuracy = accuracy_score(test_all_labels, predict)


    return accuracy, precision, recall, f1, feature_importance

if __name__ == '__main__':

    main()

