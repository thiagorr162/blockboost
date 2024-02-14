#include <bits/stdc++.h>

#define DEFAULT_NAN_STRING "-9999"

#define DEBUG 1
//#define VDEBUG 1

#ifdef DEBUG
#define dump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )
#else
#define dump(X)
#endif

#ifdef VDEBUG
#define vdump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )
#else
#define vdump(X)
#endif

using namespace std;

void load_csv(vector<double> &v, int &n, int &d, const string path) {
    ifstream f(path);
    string line;

    if (!f) {
        cerr << "error reading " << path << endl;
        exit(1);
    }
    // discard header
    getline(f, line);

    n = 0;
    d = -1;

    for (char c : line) {
        d += c== ',';
    }


    while (!f.eof()) {
        getline(f, line);

        if (line.size() > 1) {
            vector<string> cells;
            stringstream ss(line);

            while(ss.good()) {
                string cell;
                getline(ss, cell, ',');
                if (cell.size() == 0)
                    cell = DEFAULT_NAN_STRING;
                vdump(cell);
                cells.push_back(cell);
            }
            cells.pop_back();
            cells.pop_back();
            if (cells.size() != d) {
                cerr << "dimensions of entry (" << cells.size() << ") does not match dimension of header (" << d << ")" << endl;
                exit(1);
            }

            for (string cell : cells) {
                v.push_back(stod(cell));
            }
            n++;
        }
        vdump(line);
    }
}


int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false); // disable legacy IO to improve performance

    if (argc != 3 && argc != 4) {
        cerr << "Usage: similarity_measure DATASET_A DATASET_B [DIMENSION]" << endl;
        cerr << endl;
        cerr << "   DIMENSION (optional): only use first DIMENSION dimensions" << endl;
        exit(1);
    }

    string path_A = argv[1];
    string path_B = argv[2];

    dump(path_A);
    dump(path_B);

    vector<double> A, B;
    int nA, nB;
    int dA, dB;

    load_csv(A, nA, dA, path_A);
    load_csv(B, nB, dB, path_B);

    int d;

    if (argc == 4) {
        d = stoi(argv[3]);

        if ((d > dA) || (d > dB)) {
            if (d > dA)
                cerr << "DIMENSION parameter (" << d << ") is bigger than dimension of A (" << dA << ")" << endl;
            if (d > dB)
                cerr << "DIMENSION parameter (" << d << ") is bigger than dimension of B (" << dB << ")" << endl;
            exit(1);
        }
    }


    if (dA != dB && argc != 4) {
        cerr << "dimension of A (" << dA << ") does not match dimension of B (" << dB << ")" << endl;
        exit(1);
    }

    if (argc != 4) {
        d = dA;
    }

    if (nB > nA) {
        swap( A,  B);
        swap(nA, nB);
    }

    vector<double> vA;
    for (int i=0; i<nA; i++) {
        double mA = 0;
        for (int j=0; j<d; j++) {
            mA += A[i*d + j] * A[i*d + j];
        }
        vA.push_back(sqrt(mA));
    }


    vector<double> vB;
    for (int i=0; i<nB; i++) {
        double mB = 0;
        for (int j=0; j<d; j++) {
            mB += B[i*d + j] * B[i*d + j];
        }
        vB.push_back(sqrt(mB));
    }

    double cAB = 0;
    for (int ia=0; ia<nA; ia++) {
        for (int ib=0; ib<nB; ib++) {
            double mAB = 0;
            for (int j=0; j<d; j++) {
                mAB += A[ia*d + j] * B[ib*d + j];
            }
            cAB += mAB / (vA[ia] * vB[ib]);
        }
    }

    cout << cAB / (nA * nB);
}
