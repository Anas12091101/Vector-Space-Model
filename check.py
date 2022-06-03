from nltk.stem import WordNetLemmatizer,PorterStemmer

lem = WordNetLemmatizer()
stem = PorterStemmer()


print(lem.lemmatize('Weaknesses'))
print(stem.stem("Weaknesses"))
