#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    int64_t n;
    int hash_size;
    cin>>n;
    cin>>hash_size;

    cout << "s\t"<< n << "\t" << hash_size << "\t" << 0 << "\t" << endl;

    for (int64_t i =0 ; i<n; i++) {
        for (int j=0; j<hash_size+1; j++) {
            cout << j << "\t";
        }
        cout << endl;
    }
}
