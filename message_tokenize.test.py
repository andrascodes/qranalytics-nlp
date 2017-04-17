import unittest

from message_tokenize import tokenize_and_stem, tokenize_only

class TestPredictSentiment(unittest.TestCase):

    # tokenize_only: 
    # should remove nouns such as apple and iphone
    # should remove numbers
    def test_tokenize_only(self):
        self.assertEqual(
            tokenize_only('This apple iphone 6 is going to be a lovely present.'), 
            ['this', 'is', 'going', 'to', 'be', 'a', 'lovely', 'present']
        )

    # tokenize_and_stem: 
    # should remove nouns such as apple and iphone
    # should remove numbers
    # should stem the words
    def test_tokenize_and_stem(self):
        self.assertEqual(
            tokenize_and_stem('This apple iphone 6 is going to be a lovely present.'),
            ['this', 'is', 'go', 'to', 'be', 'a', 'love', 'present']
        )

if __name__ == '__main__':
    unittest.main()