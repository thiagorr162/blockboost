#!/bin/bash

parallel -j1 'bin/models/blockboost/train restaurant-ctt_3 0 1 {} 70' ::: \
    1 11 21 31 41 51 61 71 81 91 101

parallel -j1 'bin/models/blockboost/train-omp restaurant-ctt_3 0 1 {} 70' ::: \
    1 11 21 31 41 51 61 71 81 91 101
