# linreg/model.py
# Tester, doesn't do anything!

class DummyModel:
    def __init__(self):
        self.w1 = 123.45
        self.w2 = 543.21
        self.bias = 10000.0

    def predict(self, square_footage: float, bedrooms: int) -> float:
        return self.w1 * square_footage + self.w2 * bedrooms + self.bias
