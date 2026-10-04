class LemerGenerator:
    def __init__(self, seed: int = 1, a: int = 423221337, m: int = 2**31) -> None:
        self.R = seed
        self.a = a
        self.m = m

    def next_random(self) -> float:
        self.R = (self.a * self.R) % self.m
        return self.R / self.m
