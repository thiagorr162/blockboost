#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false); // Disable legacy IO to improve performance

    random_device rd;
    mt19937_64 rnd(rd());
    rnd.seed(0);

    uint64_t n = stoull(argv[1]);
    int dv = 300;

    uniform_real_distribution<double> d(0, 1);

    vector<float> v(300*n);


    cerr << "initializing string..." << endl;
    stringstream ss;
    // initialize vectorization
    for (uint64_t i=0; i<n; i++) {
        for (uint64_t j=0; j<dv; j++) {
            ss << d(rnd) << ",";
        }
    }

    
    //char line_str[1024*1024];
    string s = ss.str();
    char * text = &s[0];

    cerr << "executing..." << endl;
    auto start = high_resolution_clock::now();


    const char sep[2] = ",";
    const char* cell = strtok(text, sep);
    float sum = 0;
    while( cell != NULL) {
        sum+= atof(cell);
        cell = strtok(NULL,sep);
    }
    auto end = high_resolution_clock::now();

    double duration = (double)duration_cast<milliseconds>(end - start).count();
    cerr << sum << endl;
    cout << duration << "ms" << endl;
}
