from nltk.corpus import names
from sklearn.model_selection import train_test_split

def extract_features(name):
    """
    Get the features used for name classification
    """
    return {
        'last_letter': name[-1]
    }

# get the names
boy_names = names.words('male.txt')
girl_names = names.words('female.txt')

# build the dataset
boy_names_dataset = [(extract_features(boy_names), 'boy') for name in boy_names]
girl_names_dataset = [(extract_features(girl_names), 'girl') for name in girl_names]

# combine datsets
data = boy_names_dataset + girl_names_dataset

# split the data in features and classes
x, y = list(zip(*data))

# split and randomize
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, shuffle=True)
print(x_train[:3])
print(y_train[:3])

