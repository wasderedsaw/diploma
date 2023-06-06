import numpy as np

from keras.utils import Sequence

# TODO
class RandomSequence(Sequence):
    def __init__(self, length: int, batch_size: int):
        self.length = length
        self.batch_size = batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_list, mask_list = [], []

        for _ in range(self.batch_size):
            image_list.append(np.random.randn(224, 224, 3))
            mask_list.append(np.random.randn(1, 224, 224))

        return image_list, mask_list

    def __iter__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item

    def on_epoch_end(self):
        pass


if __name__ == '__main__':
    for i, (img, mask) in zip(range(3), RandomSequence(3, 1)):
        print(img)