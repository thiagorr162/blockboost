# reference: https://github.com/AlanConstantine/CanopyByPython

import math
import random
from datetime import datetime
from pprint import pprint as p

import matplotlib.pyplot as plt
import numpy as np


def euclideanDistance(vec1, vec2):
    return math.sqrt(((vec1 - vec2) ** 2).sum())


def jaccard(vec1, vec2, vec_to_set=True):

    if vec_to_set:
        setA, setB = set(vec1), set(vec2)

    return 1 - len(setA.intersection(setB)) / len(setA.union(setB))


def jaccardMeanDistance(vec1, vec2):
    n_features = len(vec1)
    dist = 0

    for i in range(n_features):

        setA = set(str(vec1[i]).split(" "))
        setB = set(str(vec2[i]).split(" "))

        dist_i = jaccard(setA, setB)
        dist += dist_i

    return dist / n_features


def jaccardConcatenated(vec1, vec2):

    vec1 = [str(el) for el in vec1]
    vec2 = [str(el) for el in vec2]

    strA = " ".join(vec1).split(" ")
    strB = " ".join(vec2).split(" ")

    return jaccard(set(strA), set(strB))


class Canopy:
    def __init__(self, dataset, distance_type, seed):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0
        self.distance_type = distance_type
        self.seed = seed

        if distance_type == "euclidean":
            self.distance = euclideanDistance
        elif distance_type == "jaccard-mean":
            self.distance = jaccardMeanDistance
        elif distance_type == "jaccard-concat":
            self.distance = jaccardConcatenated
        elif distance_type == "jaccard":
            self.distance = jaccard
        else:
            raise Exception("Distance not implemented")

    def setThreshold(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):

        print(len(self.dataset))

        random.seed(self.seed)

        if self.t1 == 0:
            print("Please set the threshold.")
        else:
            canopies = []

            while len(self.dataset) > 1:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]
                current_center_list = []
                delete_list = []
                self.dataset = np.delete(self.dataset, rand_index, 0)
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.distance(current_center, datum)

                    if distance <= self.t1:
                        current_center_list.append(datum)
                    if distance <= self.t2:
                        delete_list.append(datum_j)

                self.dataset = np.delete(self.dataset, delete_list, 0)
                canopies.append((current_center, current_center_list))

                print(f"Remaining elements: {len(self.dataset)}")
        return canopies
