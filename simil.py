from pyphonetics import Soundex
from pyphonetics import Metaphone

def get_levenshtein_distance(word1, word2):
    """
    https://en.wikipedia.org/wiki/Levenshtein_distance
    :param word1:
    :param word2:
    :return:
    """
    word2 = word2.lower()
    word1 = word1.lower()
    matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]

    for x in range(len(word1) + 1):
        matrix[x][0] = x
    for y in range(len(word2) + 1):
        matrix[0][y] = y

    for x in range(1, len(word1) + 1):
        for y in range(1, len(word2) + 1):
            if word1[x - 1] == word2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )

    return matrix[len(word1)][len(word2)]


soundex = Soundex()
metaphone = Metaphone()
print(soundex.phonetics("Cancer"))
print(soundex.phonetics("Cancel"))
print(soundex.distance("Cancer","Cancel"))
print(get_levenshtein_distance("Cancer","Cancel"))
print(get_levenshtein_distance(soundex.phonetics("Cancer"),soundex.phonetics("Cancel")))


print(metaphone.phonetics("Minnesota"))
print(metaphone.phonetics("Minneapolis"))
print(metaphone.distance("Minneapolis","Minnesota"))
print(get_levenshtein_distance("Minneapolis","Minnesota"))
print(get_levenshtein_distance(soundex.phonetics("Minneapolis"),soundex.phonetics("Minnesota")))

