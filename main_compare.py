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

    print("Bag of N-Grams")
    extract_bong_features(train_file, test_file, comment_file, args)

    print("Emotion")
    # extract_emotion_features(train_file, test_file, comment_file, args)

    print("Emotion and Sentiment")
    # extract_sentiment(train_file, test_file, comment_file)
