from sentiment_data import WordEmbeddings, read_word_embeddings, SentimentExample, read_sentiment_examples, read_blind_sst_examples, write_sentiment_examples

#WordEmbeddings = WordEmbeddings()
WordEmbeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
word_index= WordEmbeddings.word_indexer
print(word_index.index_of("只"))
"""
SentimentExample = read_sentiment_examples("data/train.txt")
count = 0
for ex in SentimentExample:
    count += 1
    print(f"Example {count}: {ex.words}")
"""