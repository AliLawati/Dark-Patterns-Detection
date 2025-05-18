# importing necessary libraries
import unittest
from tfidf_classification import classify_with_tf_idf


# creating a tester class for running the tests
class TfIdfTester(unittest.TestCase):

    # method for "Safe" classification test case
    def test_classify_with_tfidf_safe(self):
        policy = ("While using our website, all of your personal data will be encrypted and secured with the highest "
                  "measures possible.")
        # assert that running the "classify_with_tf_idf" with the given policy returns "Safe" classification
        self.assertEquals(classify_with_tf_idf(policy), "Safe")

    # method for "Dark" classification test case
    def test_classify_with_tfidf_dark(self):
        policy = "In some circumstances, your personal data might be sold to third-parties."
        # assert that running the "classify_with_tf_idf" with the given policy returns "Dark" classification
        self.assertEquals(classify_with_tf_idf(policy), "Dark")


# running the test scripts
if __name__ == "__main__":
    unittest.main()
