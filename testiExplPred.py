import unittest
import numpy as np
from explpred import ExplainPredictions
from sklearn.ensemble import RandomForestClassifier


class ExpPredTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X = np.random.choice([0, 1], (10000, 3))
        self.y = np.logical_xor(self.X[:, 0], self.X[:, 1])*1
        self.clf = RandomForestClassifier(max_depth=10, max_features=None)

    def test_data_preparation(self):
        # check if data is transformed
        pass

    def test_xor(self):
        return
        test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        model = self.clf.fit(self.X[:, :2], self.y)
        e = ExplainPredictions(model, self.X[:, :2], self.y)
        result = self.__get_average_sc(test, e)
        np.testing.assert_array_almost_equal(result, np.asarray(
            [[-0.25, -0.25], [0.25, 0.25], [0.25, 0.25], [-0.25, -0.25]]), decimal=1)

    def test_with_uninformative(self):
        test = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [
                        0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        model = self.clf.fit(self.X, self.y)
        e = ExplainPredictions(model, self.X, self.y)
        result = self.__get_average_sc(test, e)
        #np.testing.assert_True(result, test, rtol=1e-1)
         np.testing.assert_array_almost_equal(result, np.asarray([[-0.25, -0.25, 0], [0.25, 0.25, 0], [0.25, 0.25, 0], [-0.25, -0.25, 0],
                                                                [-0.25, -0.25, 0], [0.25, 0.25, 0], [0.25, 0.25, 0], [-0.25, -0.25, 0]]), decimal=1)

    def __get_average_sc(self, test, expPred, num_rep=2):
        accum_arr = np.zeros(test.shape)
        for i in range(num_rep):
            e = expPred.anytime_explain(test)
            accum_arr += e

        return (accum_arr / num_rep)


if __name__ == "__main__":
    unittest.main()
