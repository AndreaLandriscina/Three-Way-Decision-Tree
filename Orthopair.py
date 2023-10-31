import copy

class Orthopair:
    def setBnd(self, bnd):
        tmp1 = set(bnd)
        tmp1 = tmp1.intersection(self.p)
        tmp2 = set(bnd)
        tmp2 = tmp2.intersection(self.n)
        if len(tmp1) != 0 or len(tmp2) != 0:
            raise ValueError("The sets are non-disjoint")
        self.bnd = bnd

    def setP(self, p):
        tmp1 = set(p)
        tmp1 = tmp1.intersection(self.n)
        tmp2 = set(p)
        tmp2 = tmp2.intersection(self.bnd)
        if len(tmp1) != 0 or len(tmp2) != 0:
            raise ValueError("The sets are non-disjoint")
        self.p = p

    def setN(self, n):
        tmp1 = set(n)
        tmp1 = tmp1.intersection(self.p)
        tmp2 = set(n)
        tmp2 = tmp2.intersection(self.bnd)
        if len(tmp1) == 0 or len(tmp2) == 0:
            raise ValueError("The sets are non-disjoint")
        self.n = n

    def __init__(self, orthopair=None, p=None, bnd=None, n=None):
        if orthopair == None:
            self.n = n
            tmp = set(p)
            tmp = tmp.intersection(self.n)
            if len(tmp) != 0:
                raise ValueError("Sets are non-disjoint")
            self.p = p
            self.setBnd(bnd)
        else:
            self.p = orthopair.p
            self.bnd = orthopair.bnd
            self.n = orthopair.n
    
    def intersect(self, orthopair):
        if sorted(orthopair.getUniverse()) != sorted(self.getUniverse()):
            raise ValueError("Different universes")

        result = copy.deepcopy(self)
        tmp = copy.deepcopy(orthopair)

        y = result.p.intersection(orthopair.p)
        result.p.clear()
        result.p.update(y)
        result.n.update(tmp.n)
        y = [value for value in result.bnd if value not in result.p]
        result.bnd.clear()
        result.bnd.update(y)
        y = [value for value in tmp.bnd if value not in result.p]
        tmp.bnd.clear()
        tmp.bnd.update(y)
        result.bnd.update(tmp.bnd)
        y = [value for value in result.bnd if value not in result.n]
        result.bnd.clear()
        result.bnd.update(y)
        return result

    def getUniverseSize(self):
        return len(self.p) + len(self.n) + len(self.bnd)

    def getUniverse(self):
        result = list()
        result.extend(self.p)
        result.extend(self.n)
        result.extend(self.bnd)
        return result

    def entropy(self):
        return len(self.bnd)/self.getUniverseSize()

    def equals(self, orthopair):
        return self.p == orthopair.p and self.bnd == orthopair.bnd and self.n == orthopair.n

    def isEmpty(self):
        return self.n == self.getUniverse()

    def getUpperSize(self):
        return len(self.p) + len(self.bnd)

    def getLowerSize(self):
        return len(self.p)

    def __str__(self):
        return f"P={self.p},\nN={self.n},\nBND={self.bnd}"
