import unittest

from predict_conversation_cluster import predict_cluster

class TestPredictCluster(unittest.TestCase):

    def test_output(self):
        output = predict_cluster('Test message text')
        self.assertTrue(isinstance(output[0], int))
        self.assertTrue(isinstance(output[1], list))
        self.assertEqual(len(output[1]), 6)

if __name__ == '__main__':
    unittest.main()