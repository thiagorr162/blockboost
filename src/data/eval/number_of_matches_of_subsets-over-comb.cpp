#include <bits/stdc++.h>

// Count the average number of matches for a given subset size.
using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage:"<< endl<<"<program> <number-of-iterations>" << endl;
        return 1;
    }

    int ITERATIONS = stoi(argv[1]);
    vector<int> v;
    int x;
    while(cin>>x) {
        v.push_back(x);
    }

    // initialize with seed zero
    srand(0);
    auto r = [](int i) {
        return rand()%i;
    };
    int step = max(1, (int)floor((double)size(v) * .01));
    vector<int64_t> ov(size(v) / step);

    for (int k=step; k < size(v); k+=step) {
        for (int it=0; it<ITERATIONS; it++) {
            random_shuffle(v.begin(), v.end(), r);
            for (int i=0;i<k; i++) {
                for (int j=i+1;j<k; j++) {
                    ov[k/step -1] += (v[i] == v[j]);
                }
            }
        }
    }
    for (int i = 1; i < size(ov)-1; i++) {
        double x = ov[i];
        int k = (i+1)*step;
        double matches = 2.0 * (double) x / (double) ITERATIONS;
        double comb_k_2 = (double)(k-1) * (double)k * .5;
        cout << matches  /comb_k_2 << endl;
    }
}
