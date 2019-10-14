import nltk
import random
from nltk.corpus import names

def extract_features(name):
    """get the features used for name classification
    added new key for number if vowels in word
    """
    return {
        'last_letter': name[-1],
        'vowel_count': len([c for c in name if c in 'AEIOUaeiou'])
    }

# get the names
boy_names = names.words('male.txt')
girl_names = names.words('female.txt')

# build the dataset
boy_names_dataset = [(extract_features(name), 'boy') for name in boy_names]
girl_names_dataset = [(extract_features(name), 'girl') for name in girl_names]

# put all the names together
data = boy_names_dataset + girl_names_dataset
print(data[-1])

# shuffle everything
random.shuffle(data)

# take a look at the data
#print(data[:5])

# split the dataset into training data and test data
cuttoff = int(0.75 * len(data))
train_data, test_data = data[:cuttoff], data[cuttoff:]

# train nltk classifier
name_classifier = nltk.DecisionTreeClassifier.train(train_data)

# print data from the classifier
# print(name_classifier.classify(extract_features('Bono')))
# print(name_classifier.classify(extract_features('lani')))
# print(name_classifier.classify(extract_features('natalie')))
# print(name_classifier.classify(extract_features('joey')))
# print(name_classifier.classify(extract_features('kester')))

# test the accuracy= correctly_labelled_samples / all_samples
print(nltk.classify.accuracy(name_classifier, test_data))

# look at the tree
print(name_classifier.pretty_format())