import unittest

from conversation_tokenize import tokenize_and_stem, tokenize_only, get_feature_token

class TestPredictSentiment(unittest.TestCase):

    # tokenize_only: 
    # should remove numbers
    def test_tokenize_only(self):
        self.assertEqual(
            tokenize_only('This apple iphone 6 is going to be a lovely present.'), 
            ['this', 'apple', 'iphone', 'is', 'going', 'to', 'be', 'a', 'lovely', 'present']
        )

    # tokenize_and_stem:
    # should remove numbers
    # should stem the words
    def test_tokenize_and_stem(self):
        self.assertEqual(
            tokenize_and_stem('This apple iphone 6 is going to be a lovely present.'),
            ['this', 'appl', 'iphon', 'is', 'go', 'to', 'be', 'a', 'love', 'present']
        )

if __name__ == '__main__':
    unittest.main()