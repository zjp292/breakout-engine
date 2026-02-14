import pickle


class Engine:
    def __init__(
        self,
    ):
        pass

    def load_pickle(self, file):
        with open(file, "rb") as f:
            return pickle.load(file)

    def create_features(self):
        pass
