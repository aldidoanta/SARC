# Import module
from nrclex import NRCLex
from eval import parse, preprocessing, print_evaluation_result
from sklearn.linear_model import LogisticRegressionCV as LogitCV
from sklearn.metrics import precision_score, recall_score, f1_score
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

    all_resp_train_acc, all_resp_test_acc, ori_train_acc, ori_test_acc, precision, recall, f1 = extract_emotion_features(train_file, test_file, comment_file, args)
    print_evaluation_result(all_resp_train_acc, all_resp_test_acc, ori_train_acc, ori_test_acc, precision, recall, f1)

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
    clf = LogitCV(Cs=[10**i for i in range(-2, 3)], fit_intercept=False, cv=2, dual=np.less(*train_all_docs_emo.shape), solver='liblinear', n_jobs=-1, random_state=0)
    clf.fit(train_all_docs_emo, train_all_labels)
    all_resp_train_acc = clf.score(train_all_docs_emo, train_all_labels)
    all_resp_test_acc = clf.score(test_all_docs_emo, test_all_labels)
    predict = clf.predict(test_all_docs_emo)

    # Get vectors for first and second responses.
    n_tr = int(train_all_docs_emo.shape[0]/2)
    n_te = int(test_all_docs_emo.shape[0]/2)
    train_vecs = {i: train_all_docs_emo[i*n_tr:(i+1)*n_tr,:] for i in range(2)}
    test_vecs = {i: test_all_docs_emo[i*n_te:(i+1)*n_te,:] for i in range(2)}

    # Final evaluation.
    hyperplane = clf.coef_[0,:]
    train_pred_labels = 2*(train_vecs[0].dot(hyperplane) > train_vecs[1].dot(hyperplane))-1
    test_pred_labels = 2*(test_vecs[0].dot(hyperplane) > test_vecs[1].dot(hyperplane))-1
    train_expect_labels = train_labels[0]
    test_expect_labels = test_labels[0]
    ori_train_acc = (train_pred_labels == train_expect_labels).sum() / train_pred_labels.shape[0]
    ori_test_acc = (test_pred_labels == test_expect_labels).sum() / test_pred_labels.shape[0]

    # Measure Performance
    precision = precision_score(test_all_labels, predict)
    recall = recall_score(test_all_labels, predict)
    f1 = f1_score(test_all_labels, predict)

    return all_resp_train_acc, all_resp_test_acc, ori_train_acc, ori_test_acc, precision, recall, f1
def extract_emo(sentence):
    emo_count = {"fear": 0, "anger": 0, "anticipation": 0, "trust": 0, "surprise": 0, "positive": 0, "negative": 0, "sadness": 0, "disgust": 0, "joy": 0}
    for word in sentence:
        emotion = NRCLex(word).raw_emotion_scores

        for emo in emotion:
            if emotion[emo] > 0:
                emo_count[emo] += 1

    return emo_count

def average_emo(prev_emo, next_emo, len_prev, len_next):
    return [prev_emo["fear"]/len_prev, prev_emo["anger"]/len_prev, prev_emo["anticipation"]/len_prev, prev_emo["trust"]/len_prev, prev_emo["surprise"]/len_prev, prev_emo["positive"]/len_prev, prev_emo["negative"]/len_prev, prev_emo["sadness"]/len_prev, prev_emo["disgust"]/len_prev, prev_emo["joy"]/len_prev, next_emo["fear"]/len_next, next_emo["anger"]/len_next, next_emo["anticipation"]/len_next, next_emo["trust"]/len_next, next_emo["surprise"]/len_next, next_emo["positive"]/len_next, next_emo["negative"]/len_next, next_emo["sadness"]/len_next, next_emo["disgust"]/len_next, next_emo["joy"]/len_next]

if __name__ == '__main__':

    main()