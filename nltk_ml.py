import nltk
import collections
from nltk.corpus import names

# check the girl name
girl_names = names.words('female.txt')
print(girl_names[:10], '...')
print('#GirlNames= ', len(girl_names))

# check the boy name
boy_names = names.words('male.txt')
print(boy_names[:10], '...')
print('#BoyNames= ', len(boy_names))

# how many girl names end with a
girl_names_ending_in_a = [name for name in girl_names if name.endswith('a')]
print('#GirNamesEndingInA=', len(girl_names_ending_in_a))

# what are the most common letters girl names end with
girl_ending_letters = collections.Counter([name[-1] for name in girl_names])
print("MostCommonEndingLettersForGirls=", girl_ending_letters)

# what are the most common last letters in boys names
boy_names_ending_letters = collections.Counter([name[-1] for name in boy_names])
print("MostCommonEndingLettersForBoys=", boy_names_ending_letters)