from eval import extract_bong_features, parse
from emotion_extraction import extract_emotion_features
from turn_level_sentiment import extract_sentiment
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
    results = []

    results.append(extract_bong_features(train_file, test_file, comment_file, args))
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

        print(model, results[row][3], "\t", results[row][4], "\t", results[row][5], "\t", results[row][6], "\t")





main()