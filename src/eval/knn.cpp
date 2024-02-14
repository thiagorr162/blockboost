#include <bits/stdc++.h>
#include <random>

using namespace std;

#define MAX_K 1024


ifstream::pos_type file_size(const string &filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

#define sdump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )

#ifdef DEBUG
    #define dump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )
#else
    #define dump(X) ;
#endif

void load_raw_double(string path, vector<double> &v) {
    basic_ifstream<char> f(path, ios::binary);
    vector<char> cv((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    v.resize(cv.size()/8);
    memcpy(&v[0], &cv[0], cv.size());
}

template <typename t>
inline int dist(t begin_0, t end_0, t begin_1) {
    int sum = 0;
    for (int i = 0; i< end_0-begin_0; i++) {
        sum += begin_0[i] != begin_1[i];
    }
    return sum;
}

template <typename t>
inline double dist_ponderated(t begin_0, t end_0, t begin_1, const vector<double> &alphas) {
    double sum = 0.0;
    int n = end_0 - begin_0;

    // sum in reverse to improve numerical stability
    for (int i = 0; i< n; i++) {
        int i_inv = n-i-1;
        sum += (double)(begin_0[i_inv] != begin_1[i_inv]) * alphas[i_inv];
    }

    return sum;
}

template <typename t>
int dist(t begin_0, t end_0, t begin_1, const int &word_size) {
    int sum = 0;
    int n_words = (end_0 - begin_0) / word_size;
    for (int i = 0; i < n_words; i++) {
        int word_diff = 0;
        for (int j=0; j<word_size; j++)
            word_diff += begin_0[i*word_size + j] != begin_1[i*word_size + j];
        if (word_diff == 0) {
            return 0;
        }
        sum += !!word_diff;
    }
    return sum;
}

string get_extension_from_path(const string &path) {
    stringstream ss(path);
    string ext;
    char dot = '.';
    while (!ss.eof()) {
        getline(ss, ext, dot);
    }
    return ext;
}

string get_dataset_path(const string &path) {
    stringstream ss(path);
    string basename;
    char dot = '/';
    while (!ss.eof()) {
        getline(ss, basename, dot);
    }

    string hparams_path = path.substr(0, path.length() - basename.length()) + "hparams.json";

    ifstream hparams_file(hparams_path);

    string dataset;
    int i = 0;
    while (!hparams_file.eof() && i++ < 4) {
        getline(hparams_file, dataset, '"');
    }

    string dataset_path;

    if (basename.find("test") == 0) {
        dataset_path = "research/data/processed/"+dataset+"/test-vectorized.csv";
    }
    if (basename.find("val") == 0) {
        dataset_path = "research/data/processed/"+dataset+"/val-vectorized.csv";
    }
    if (basename.find("train") == 0) {
        dataset_path = "research/data/processed/"+dataset+"/train-vectorized.csv";
    }
    return dataset_path;
}


tuple<vector<int>, vector<int>>  get_index_2_ei_ti(const string &dataset_path, int &n_tables) {
    map<string, int> entity_id_2_ei;
    map<string, int> head_record_id_2_ti;

    vector<int> index_2_ei;
    vector<int> index_2_ti;

    ifstream dataset_file(dataset_path, ios::in);
    cerr << endl << "dataset_lines" << endl;

    // jump header
    string line;
    getline(dataset_file, line);

    int index = 0;

    while (!dataset_file.eof()) {
        getline(dataset_file, line);

        if (line.size() == 0)
            break;

        stringstream ss(line);
        vector<string> ve;
        while (!ss.eof()) {
            string entry;
            getline(ss, entry, ',');
            ve.push_back(entry);
        }
        string record_id = ve[ve.size()-1];
        string entity_id = ve[ve.size()-2];
        string head_record_id = record_id.substr(0, 4);


        if (entity_id_2_ei.find(entity_id) == entity_id_2_ei.end())
            entity_id_2_ei[entity_id] = entity_id_2_ei.size();

        if (head_record_id_2_ti.find(head_record_id) == head_record_id_2_ti.end())
            head_record_id_2_ti[head_record_id] = head_record_id_2_ti.size();

        int ti = head_record_id_2_ti[head_record_id];
        int ei = entity_id_2_ei[entity_id];

        index_2_ti.push_back(ti);
        index_2_ei.push_back(ei);

        index++;
    }

    n_tables = head_record_id_2_ti.size();

    return {index_2_ei, index_2_ti};
}

void load_emb(string &input_path, int &n_tables, vector<uint64_t> &vec_n_entries, int &hash_size, vector<double> &alphas, vector<vector<int>> &vv) {
    string shape_path = input_path.substr(0, input_path.length()-3 ) + "shape";
    string alphas_path = input_path.substr(0, input_path.length()-3 ) + "alphas";

    ifstream shape_file(shape_path, ios::in);
    ifstream alphas_file(alphas_path, ios::binary);
    ifstream input_file(input_path, ios::binary);


    int total_n_entries;
    int word_size;
    shape_file >> total_n_entries;
    string dataset_path = get_dataset_path(input_path);

    int64_t input_size = file_size(input_path);

    hash_size = input_size / total_n_entries;


    load_raw_double(alphas_path, alphas);

    // sum in reverse order, to improve numerical stability
    double acc_alphas = 0.0;
    for (int i=0; i<hash_size; i++) {
        acc_alphas += alphas[hash_size-1-i];
    }

    for (int i=0; i<hash_size; i++) {
        alphas[i] = alphas[i]/acc_alphas;
    }

    dump(total_n_entries);
    dump(hash_size);

    auto [index_2_ei, index_2_ti] = get_index_2_ei_ti(dataset_path, n_tables);
    int count = 0;

    dump(n_tables);

    // initialize empty tables
    for (int ti=0; ti < n_tables; ti++) {
        vv.push_back({});
        vec_n_entries.push_back(0);
    }

    for (int i=0; i<total_n_entries; i++) {
        int ei = index_2_ei[i];
        int ti = index_2_ti[i];
        vec_n_entries[ti]++;

        vv[ti].push_back(ei);

        for (int j=0; j<hash_size; j++) {
            if (input_file.eof()) {
                cerr << "error: unexpedted end of file." << endl;
                dump(input_path);
                dump(input_file.eof());
                exit(1);
            }
            int8_t x;
            input_file >> x;
            vv[ti].push_back(x);
        }
    }
}

int main(int argc, char* argv[]) {
    cout.precision(numeric_limits<double>::max_digits10); // make cout precise
    ios::sync_with_stdio(false); // Disable legacy IO to improve performance

    if (argc != 2) {
        cerr << "usage: " << endl;
        cerr << "  knn PATH_TO_INPUT" << endl;
        cerr << endl;
        cerr << "the output of the program is:" << endl;
        cerr << "tp" << endl;
        cerr << "tn" << endl;
        cerr << "fp" << endl;
        cerr << "fn" << endl;
        return -1;
    }

    string input_path = argv[1];

    string extension = get_extension_from_path(input_path);


    // input variables
    int n_tables;
    vector<uint64_t> vec_n_entries = {};
    int hash_size;
    vector<double> alphas;
    vector<vector<int>> vv;
    int max_dist;
    int word_size = 1;


    if (extension != "emb" ) {
        cerr << "error: embedding must be an .emb file" << endl;
        exit(1);
    }

    load_emb(input_path, n_tables, vec_n_entries, hash_size, alphas, vv);

    uint64_t min_table_size = *min_element(vec_n_entries.begin(), vec_n_entries.end());
    uint64_t num_hparams = min(min_table_size, (uint64_t)MAX_K);

    // output variables
    vector<uint64_t> tp(num_hparams);
    vector<uint64_t> tn(num_hparams);
    vector<uint64_t> fp(num_hparams);
    vector<uint64_t> fn(num_hparams);

    // output variables (ponderated by alphas)
    vector<uint64_t> tp_a(num_hparams);
    vector<uint64_t> tn_a(num_hparams);
    vector<uint64_t> fp_a(num_hparams);
    vector<uint64_t> fn_a(num_hparams);

    dump(hash_size);

    mt19937 gen;
    gen.seed(0);

    if (n_tables > 1) {
        for (int ti=0; ti<n_tables; ti++) {
            for (int tj=ti+1; tj<n_tables; tj++) {


                uint64_t step = 1 + (vec_n_entries[tj] - 1);
                uint64_t j_f = min(step, vec_n_entries[tj]);

                for (int i=0; i<vec_n_entries[ti]; i++) {
                    auto begin_0 = vv[ti].begin()+(i*(hash_size*word_size+1))+1;
                    auto end_0 = begin_0 + hash_size*word_size;

                    vector<pair<int, int>> vd;
                    vector<pair<double, int>> vd_a;

                    for (uint64_t j=0; j<j_f; j++) {
                        auto begin_1 = vv[tj].begin()+j*(hash_size*word_size+1)+1;
                        int d = dist(begin_0, end_0, begin_1);
                        double d_a = dist_ponderated(begin_0, end_0, begin_1, alphas);
                        vd.push_back(make_pair(d, vv[tj][j*(hash_size*word_size+1)]));
                        vd_a.push_back(make_pair(d_a, vv[tj][j*(hash_size*word_size+1)]));
                    }

                    shuffle(vd.begin(), vd.end(), gen);
                    shuffle(vd_a.begin(), vd_a.end(), gen);

                    sort(vd.begin(), vd.end(), [](auto a, auto b){return get<0>(a) < get<0>(b);});
                    sort(vd_a.begin(), vd_a.end(), [](auto a, auto b){return get<0>(a) < get<0>(b);});

                    for (uint64_t j=0; j<j_f; j++) {
                        //dump(j);
                        //dump(get<0>(vd_a[j]));
                        //dump(get<1>(vd_a[j]));

                        for (uint64_t k=0; k<num_hparams; k++) {
                            bool is_close = (j <= k);
                            bool gt = get<1>(vd[j]) == vv[ti][i*(hash_size*word_size+1)];
                            bool gt_a = get<1>(vd_a[j]) == vv[ti][i*(hash_size*word_size+1)];

                            tp[k] += is_close * gt;
                            tn[k] += !is_close * !gt;
                            fp[k] += is_close * !gt;
                            fn[k] += !is_close * gt;

                            bool is_close_a = (j <= k);
                            tp_a[k] += is_close_a * gt_a;
                            tn_a[k] += !is_close_a * !gt_a;
                            fp_a[k] += is_close_a * !gt_a;
                            fn_a[k] += !is_close_a * gt_a;
                        }
                    }

                }
            }
        }
    } else {

        int ti = 0;
        int tj = 0;

        uint64_t step = 1 + (vec_n_entries[tj] - 1);
        uint64_t j_f = min(step, vec_n_entries[tj]);

        for (uint64_t i=0; i<vec_n_entries[ti]; i++) {
            auto begin_0 = vv[ti].begin()+(i*(hash_size*word_size+1))+1;
            auto end_0 = begin_0 + hash_size*word_size;

            vector<pair<int, int>> vd;
            vector<pair<double, int>> vd_a;

            for (uint64_t j=0; j<j_f; j++) {

                if (i != j) {
                    auto begin_1 = vv[tj].begin()+j*(hash_size*word_size+1)+1;
                    int d = dist(begin_0, end_0, begin_1);
                    double d_a = dist_ponderated(begin_0, end_0, begin_1, alphas);
                    vd.push_back(make_pair(d, vv[tj][j*(hash_size*word_size+1)]));
                    vd_a.push_back(make_pair(d_a, vv[tj][j*(hash_size*word_size+1)]));
                }

            }

            shuffle(vd.begin(), vd.end(), gen);
            shuffle(vd_a.begin(), vd_a.end(), gen);

            sort(vd.begin(), vd.end(), [](auto a, auto b){return get<0>(a) < get<0>(b);});
            sort(vd_a.begin(), vd_a.end(), [](auto a, auto b){return get<0>(a) < get<0>(b);});

            for (uint64_t j=0; j<j_f-1; j++) {
                dump(j);
                dump(get<0>(vd_a[j]));
                dump(get<1>(vd_a[j]));

                for (uint64_t k=0; k<num_hparams; k++) {
                    bool is_close = (j <= k);
                    bool gt = get<1>(vd[j]) == vv[ti][i*(hash_size*word_size+1)];
                    bool gt_a = get<1>(vd_a[j]) == vv[ti][i*(hash_size*word_size+1)];

                    tp[k] += is_close * gt;
                    tn[k] += !is_close * !gt;
                    fp[k] += is_close * !gt;
                    fn[k] += !is_close * gt;

                    bool is_close_a = (j <= k);
                    tp_a[k] += is_close_a * gt_a;
                    tn_a[k] += !is_close_a * !gt_a;
                    fp_a[k] += is_close_a * !gt_a;
                    fn_a[k] += !is_close_a * gt_a;
                }
            }
        }
    }

    for (int md=0;md<num_hparams; md++) {
        cout << tp[md] << "\t";
        cout << tn[md] << "\t";
        cout << fp[md] << "\t";
        cout << fn[md] << endl;
    }

    for (int md=0;md<num_hparams; md++) {
        cout << tp_a[md] << "\t";
        cout << tn_a[md] << "\t";
        cout << fp_a[md] << "\t";
        cout << fn_a[md] << endl;
    }

}
