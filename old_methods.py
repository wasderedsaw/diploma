import abc

import cv2
import numpy as np


def load_base(filename):
    a = np.loadtxt(filename, np.float32, delimiter=',', skiprows=1)

    samples, responses = a[:, 1:], a[:, 0]

    return samples, responses


class OldModel:
    __metaclass__ = abc.ABCMeta

    class_n = 10
    train_ratio = 0.5
    model = None

    def load(self, filename):
        self.model.load(filename)

    def save(self, filename):
        self.model.save(filename)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n + 1), np.float32)
        new_samples[:, :-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:, -1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n * self.class_n, np.int32)
        resp_idx = np.int32(responses + np.arange(sample_n) * self.class_n)
        new_responses[resp_idx] = 1
        return new_responses

    @abc.abstractmethod
    def predict(self, samples) -> np.ndarray:
        pass

    @abc.abstractmethod
    def train(self, samples, responses) -> np.ndarray:
        pass


class RTrees(OldModel):
    def __init__(self):
        self.model = cv2.ml.RTrees_create()

    def train(self, samples, responses):
        self.model.setMaxDepth(20)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

    def predict(self, samples) -> np.ndarray:
        _ret, resp = self.model.predict(samples)
        return resp.ravel()


class KNearest(OldModel):
    def __init__(self):
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples) -> np.ndarray:
        _retval, results, _neigh_resp, _dists = self.model.findNearest(samples, k=10)
        return results.ravel()


class Boost(OldModel):
    def __init__(self):
        self.model = cv2.ml.Boost_create()

    def train(self, samples, responses):
        _sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.ml.VAR_NUMERICAL] * var_n + [cv2.ml.VAR_CATEGORICAL, cv2.ml.VAR_CATEGORICAL],
                             np.uint8)

        self.model.setWeakCount(15)
        self.model.setMaxDepth(10)
        self.model.train(
            cv2.ml.TrainData_create(new_samples, cv2.ml.ROW_SAMPLE, new_responses.astype(int), varType=var_types))

    def predict(self, samples) -> np.ndarray:
        new_samples = self.unroll_samples(samples)
        _ret, resp = self.model.predict(new_samples)

        return resp.ravel().reshape(-1, self.class_n).argmax(1)


class SVM(OldModel):
    def __init__(self):
        self.model = cv2.ml.SVM_create()

    def train(self, samples, responses):
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(1)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setGamma(.1)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

    def predict(self, samples) -> np.ndarray:
        _ret, resp = self.model.predict(samples)
        return resp.ravel()


class MLP(OldModel):
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def train(self, samples, responses):
        _sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)
        layer_sizes = np.int32([var_n, 100, 100, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.0)
        self.model.setBackpropWeightScale(0.001)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples) -> np.ndarray:
        print(samples)
        _ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


# TODO implement
class EM(OldModel):
    def __init__(self):
        self.model = cv2.ml.EM_create()

    def train(self, samples, responses):
        pass

    def predict(self, samples):
        pass

class FakeModel(OldModel):
    def train(self, samples, responses):
        pass

    def predict(self, samples) -> np.ndarray:
        return np.array(map(lambda _: 9, samples))


def train_on_csv_data(model: OldModel, path_to_data: str) -> None:
    samples, responses = load_base(path_to_data)
    train_n = int(len(samples) * model.train_ratio)
    model.train(samples[:train_n], responses[:train_n])


if __name__ == '__main__':
    pass