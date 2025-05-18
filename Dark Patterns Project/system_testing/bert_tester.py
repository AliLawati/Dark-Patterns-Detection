# importing necessary libraries
import unittest
from bert_classification import classify_with_bert


# creating a tester class for running the tests
class BertTester(unittest.TestCase):

    # method for "Safe" classification test case
    def test_classify_with_bert_safe(self):
        policy = ("While using our website, all of your personal data will be encrypted and secured with the highest "
                  "measures possible.")
        # assert that running the "classify_with_bert" with the given policy returns "Safe" classification
        self.assertEquals(classify_with_bert(policy), "Safe")

    # method for "Dark" classification test case
    def test_classify_with_bert_dark(self):
        policy = "In some circumstances, your personal data might be sold to third-parties."
        # assert that running the "classify_with_bert" with the given policy returns "Dark" classification
        self.assertEquals(classify_with_bert(policy), "Dark")


# running the test scripts
if __name__ == "__main__":
    unittest.main()
