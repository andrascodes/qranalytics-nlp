import unittest

from predict_message_sentiment import predict_sentiment

class TestPredictSentiment(unittest.TestCase):

    def test_positive(self):
        self.assertEqual(predict_sentiment('love'), 'negative')

    def test_negative(self):
        self.assertEqual(predict_sentiment('bad'), 'negative')

    def test_neutral(self):
        self.assertEqual(predict_sentiment('This is a neutral sentence.'), 'neutral')

if __name__ == '__main__':
    unittest.main()