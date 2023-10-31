import numpy as np
import Orthopair
import copy

class Orthopartition:
    def __init__(self, family=None, overlap=None):
        self.family = list()
        if overlap is None:
            self.overlap = False
            tmp = list()
            for x in family:
                tmp.append(Orthopair.Orthopair(orthopair=x))
            self.family = list(tmp)
            for x in self.family:
                for y in self.family:
                    if x != y:
                        tmp1 = copy.deepcopy(x)
                        tmp1.p.update(tmp1.bnd)
                        inters = tmp1.p.intersection(y.p)
                        tmp1.p.clear()
                        tmp1.p.update(inters)

                        tmp2 = copy.deepcopy(y)
                        inters = tmp2.p.intersection(y.bnd)
                        tmp2.p.clear()
                        tmp2.p.update(inters)
                        if len(tmp1.p) != 0 or len(tmp2.p) != 0:
                            self.overlap = True
                            break
                if self.overlap is True:
                    break
        else:
            self.overlap = overlap
            lst = copy.deepcopy(family)
            universe = lst[0].getUniverse()
            for orthopair in lst:
                if sorted(orthopair.getUniverse()) != sorted(universe):
                    raise Exception(
                        "Not all orthopairs are defined on the same universe")
            if not (overlap):
                for orthopair in lst:
                    for orthopair1 in lst:
                        if orthopair != orthopair1:
                            tmp1 = copy.deepcopy(orthopair)
                            tmp1.p.update(tmp1.bnd)
                            y = tmp1.p.intersection(orthopair1.p)
                            tmp1.p.clear()
                            tmp1.p.update(y)
                            tmp2 = copy.deepcopy(orthopair1)
                            y = tmp2.p.intersection(orthopair1.bnd)
                            tmp2.p.clear()
                            tmp2.p.update(y)
                            if len(tmp1.p) != 0 or len(tmp2.p) != 0:
                                raise Exception("Orthopairs overlap")
            self.family = family

    @staticmethod
    def totalBoundary(orthopairs):
        sum = 0
        for orthopair in orthopairs:
            sum += len(orthopair.bnd)
        return sum

    def lowerEntropy(self):
        lower_Entropy = 0
        tmp = copy.deepcopy(self.family)
        if not (self.overlap):
            while Orthopartition.totalBoundary(tmp) != 0:
                max = None
                for orthopair in tmp:
                    if max == None and orthopair.entropy() > 0:
                        max = orthopair
                    if orthopair.entropy() > 0 and orthopair.getUpperSize() > max.getUpperSize():
                        max = orthopair

                max.p.update(max.bnd)
                max.setBnd(set())

                for orthopair in tmp:
                    if orthopair != max:
                        y = [x for x in orthopair.bnd if x not in max.p]
                        orthopair.bnd.clear()
                        orthopair.bnd.update(y)
                        orthopair.n.update(max.p)

            for orthopair in tmp:
                for orthopair1 in tmp:
                    if orthopair != orthopair1:
                        lower_Entropy += orthopair.getLowerSize()*orthopair1.getLowerSize()
        else:
            while Orthopartition.totalBoundary(tmp) != 0:
                for orthopair in tmp:
                    orthopair.p.update(orthopair.bnd)
                    orthopair.setBnd(set())
            for orthopair in tmp:
                for orthopair1 in tmp:
                    if orthopair != orthopair1:
                        to = copy.deepcopy(orthopair)
                        tp = copy.deepcopy(orthopair1)
                        inters = to.intersect(tp).p
                        y = [x for x in to.p if x not in inters]
                        to.p.clear()
                        to.p.update(y)
                        lower_Entropy += to.getLowerSize()*tp.getLowerSize()
        x = lower_Entropy/np.square(tmp[0].getUniverseSize())
        return x

    def upperEntropy(self):
        tmp = copy.deepcopy(self.family)
        upperEntropy = 0
        while Orthopartition.totalBoundary(tmp) != 0:
            min = None
            for orthopair in tmp:
                if min == None and orthopair.entropy() > 0:
                    min = orthopair
                if orthopair.entropy() > 0 and orthopair.getLowerSize() < min.getLowerSize():
                    min = orthopair
            i = list(min.bnd)[0]
            min.p.add(i)
            min.bnd.remove(i)
            for orthopair in tmp:
                if orthopair != min:
                    Bnd = [value for value in orthopair.bnd if value not in min.p]
                    orthopair.setBnd(set(Bnd))
                    orthopair.n.update(min.p)
        for orthopair in tmp:
            for orthopair1 in tmp:
                if orthopair != orthopair1:
                    if not (self.overlap):
                        upperEntropy += orthopair.getLowerSize()*orthopair1.getLowerSize()
                    else:
                        to = copy.deepcopy(orthopair)
                        tp = copy.deepcopy(orthopair1)
                        inters = to.intersect(tp).p
                        y = [x for x in to.p if x not in inters]
                        to.p.clear()
                        to.p.update(y)
                        upperEntropy += to.getLowerSize()*tp.getLowerSize()
        x = upperEntropy/np.square(tmp[0].getUniverseSize())
        return x

    def meet(self, pi):
        tmp = list()
        for orthopair in self.family:
            for orthopair1 in pi.family:
                inters = orthopair.intersect(orthopair1)
                if inters.isEmpty() is False:
                    tmp.append(inters)
        overlap = self.overlap or pi.overlap
        return Orthopartition(family=tmp, overlap=overlap)

    def mutual_information(self, pi):
        result = 0
        result1 = self.family[0].getUniverseSize(
        )*(self.lowerEntropy() + self.upperEntropy()) / 2
        result2 = self.family[0].getUniverseSize(
        )*(pi.lowerEntropy() + pi.upperEntropy()) / 2
        m = self.meet(pi)
        result = result1 + result2 - (self.family[0].getUniverseSize(
        )*m.lowerEntropy() + self.family[0].getUniverseSize() * m.upperEntropy())/2
        if result1 > result2:
            result /= (result1 + 0.0001)
        else:
            result /= (result2 + 0.0001)
        return result

    def __str__(self):
        strr = ""
        for x in self.family:
            strr += x.__str__() + "\n"
        return strr
