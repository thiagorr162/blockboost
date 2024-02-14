#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

#define EPS 1e-14

template <typename T>
inline int8_t sign(const T &a) {
    return (a + EPS >= 0) - (a + EPS < 0);
}

int main(int argc, char* argv[]) {
    random_device rd;
    mt19937_64 rnd(rd());
    rnd.seed(0);

    uint64_t n = stoull(argv[1]);
    int dv = 300;
    int bits = 256;

    uniform_real_distribution<double> d(0, 1);
    uniform_int_distribution<int> di(0, dv-1);

    vector<pair<int,float>> history;


    cerr << "initializing vectors..." << endl;
    // initialize history
    for (int i=0; i<bits; i++) {
        history.push_back(make_pair(di(rnd), d(rnd)));
    }

    vector<float> v(300*n);

    // initialize vectorization
    for (uint64_t i=0; i<n; i++) {
        for (uint64_t j=0; j<dv; j++) {
            v.push_back(d(rnd));
        }
    }

    vector<int8_t> emb(bits*n);

    cerr << "executing..." << endl;

    auto start = high_resolution_clock::now();

#pragma omp parallel
{
    vector<pair<int,float>> history_p(history.begin(), history.end());

#pragma omp for schedule(static, 1024)
    for (uint64_t i=0; i<n; i++) {
        for (int j=0; j< history_p.size(); j++) {
            auto h = history_p[j];
            emb[i*history.size()+j] = sign(v[i*300 + h.first] - h.second);
        }
    }
}
    auto end = high_resolution_clock::now();
    double duration = (double)duration_cast<milliseconds>(end - start).count();
    cout << duration << "ms" << endl;
}
