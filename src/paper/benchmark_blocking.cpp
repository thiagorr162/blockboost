#include <bits/stdc++.h>

#define dump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    random_device rd;
    mt19937_64 rnd(rd());
    rnd.seed(0);
    uniform_int_distribution<uint64_t> dist(0, UINT_MAX);

    int max_hamming = stoi(argv[1]);
    int size = stoi(argv[2]);

    vector<uint64_t> v(size * 4);


    auto start = high_resolution_clock::now();
    cerr << "Initilizing array..." << endl;
    for (auto it=v.begin(); it!=v.end(); it++) {
        *it = dist(rnd);
    }
    cerr << "" << endl;
    cerr << (double) 1e-6 * duration_cast<microseconds>(high_resolution_clock::now() - start).count() << "s " << endl;;
    cerr << endl;
    cerr << "Computing hammings for max_hamming=" << max_hamming << endl;

    uint64_t a = 0;
    start = high_resolution_clock::now();

    vector<pair<int, int>> candidates;

    #pragma omp parallel
    {
        vector<pair<int, int>> t_candidates;
        uint64_t a_partial = 0;

        #pragma omp for schedule(static, 256)
        for (int i=0; i<size; i++) {
            for (int j=i+1; j<size; j++) {
                int dist;
                dist  = __builtin_popcountll(v[4*i  ] ^ v[4*j  ]);
                dist += __builtin_popcountll(v[4*i+1] ^ v[4*j+1]);
                dist += __builtin_popcountll(v[4*i+2] ^ v[4*j+2]);
                dist += __builtin_popcountll(v[4*i+3] ^ v[4*j+3]);

                if (dist <= max_hamming) {
                    t_candidates.push_back(make_pair(i, j));
                }
            }

        }

        #pragma omp critical
        {
            for (auto c : t_candidates) {
                candidates.push_back(c);
            }
        }
    }
    auto end = high_resolution_clock::now();

    dump(candidates.size());
    double duration = duration_cast<microseconds>(end - start).count();
    duration = duration / 1e6;
    cout << duration << " sec" << endl;
}
