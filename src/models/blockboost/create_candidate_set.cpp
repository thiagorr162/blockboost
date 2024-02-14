#include <bits/stdc++.h>
#include <omp.h>

using namespace std;
using namespace chrono;

ifstream::pos_type file_size(const string &filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}


#define sdump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )

#define DEBUG

#ifdef DEBUG
    #define dump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )
#else
    #define dump(X) ;
#endif

#ifdef VDEBUG
    #define vdump(X) ( cerr << __LINE__ << ": " << #X << "=" << X << endl )
#else
    #define vdump(X) ;
#endif

void load_raw_double(string path, vector<double> &v) {
    basic_ifstream<char> f(path, ios::binary);
    vector<char> cv((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    v.resize(cv.size()/8);
    memcpy(&v[0], &cv[0], cv.size());
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


tuple<vector<int>, vector<int>>  get_index_2_ei_ti(const string &dataset_path, int &n_tables, vector<string> &index_2_record_id) {
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

        index_2_record_id.push_back(record_id);
        index_2_ti.push_back(ti);
        index_2_ei.push_back(ei);

        index++;
    }

    n_tables = head_record_id_2_ti.size();

    return {index_2_ei, index_2_ti};
}


void load_emb(string &input_path, int &n_tables, vector<uint64_t> &vec_n_entries, int &hash_size, int &long_hash_size, vector<vector<uint64_t>> &vv, vector<vector<string>> &ti_2_i_2_record_id) {
    string shape_path = input_path.substr(0, input_path.length()-3 ) + "shape";

    ifstream shape_file(shape_path, ios::in);
    ifstream input_file(input_path, ios::binary);

    if (!shape_file) {
        cerr << "error opening " << shape_path << " for reading" << endl;
        exit(1);
    }
    if (!input_file) {
        cerr << "error opening " << input_path << " for reading" << endl;
        exit(1);
    }

    int total_n_entries;
    int word_size;
    shape_file >> total_n_entries;
    string dataset_path = get_dataset_path(input_path);

    int64_t input_size = file_size(input_path);

    long_hash_size = input_size / total_n_entries;
    hash_size = (long_hash_size / 64) + ( long_hash_size % 64 > 0) ;


    vector<string> index_2_record_id;
    auto [index_2_ei, index_2_ti] = get_index_2_ei_ti(dataset_path, n_tables, index_2_record_id);
    int count = 0;

    dump(n_tables);

    // initialize empty tables
    for (int ti=0; ti < n_tables; ti++) {
        vv.push_back({});
        //ve.push_back({});
        ti_2_i_2_record_id.push_back({});
        vec_n_entries.push_back(0);
    }

    for (int i=0; i<total_n_entries; i++) {
        int ei = index_2_ei[i];
        int ti = index_2_ti[i];
        vec_n_entries[ti]++;

        //ve[ti].push_back(ei);
        ti_2_i_2_record_id[ti].push_back(index_2_record_id[i]);

        uint64_t word = 0;
        for (int j=0; j<long_hash_size; j++) {
            if (input_file.eof()) {
                cerr << "error: unexpedted end of file." << endl;
                dump(input_path);
                dump(input_file.eof());
                exit(1);
            }
            int8_t x;
            input_file >> x;
            x = (x+1)/2;
            vdump(int(x));
            word = word << 1;
            word += x;


            if ((j == long_hash_size-1) || (j % 64 == 63)) {
                vdump(word);
                vv[ti].push_back(word);
                word = 0;
            }
            //vv[ti].push_back(x);
        }
    }
}

template <int HASH_SIZE>
void multi_table_loop(
        vector<pair<int, int>> &candidates,
        const int &max_hamming_distance,
        const int &n_tables,
        const int &long_hash_size,
        vector<uint64_t> &vec_n_entries,
        vector<vector<uint64_t>> &vv,
        ofstream &out
    ) {


    uint64_t counter = 0;
    for (int ti=0; ti<n_tables; ti++) {
        for (int tj=ti+1; tj<n_tables; tj++) {



            #pragma omp parallel
            {
                vector<pair<int, int>> t_candidates;
                uint64_t a_partial = 0;


                uint64_t* v_left = &vv[ti][0];
                uint64_t* v_right = &vv[tj][0];

                #pragma omp for schedule(static, 256)
                for (int i=0; i<vec_n_entries[ti]; i++) {
                    uint64_t vi0 = HASH_SIZE*i;

                    for (int j=0; j<vec_n_entries[tj]; j++) {
                        uint64_t vj0 = HASH_SIZE*j;
                        int dist;
                        dist  = __builtin_popcountll(v_left[vi0  ] ^ v_right[vj0  ]);
                        for(int k=1; k<HASH_SIZE; k++) {
                            dist  += __builtin_popcountll(v_left[vi0+k] ^ v_right[vj0+k]);
                        }

                        if (dist <= max_hamming_distance) {
                            t_candidates.push_back(make_pair(i, j));
                        }
                    }

                }

                #pragma omp critical
                {
                    out.write((char*)&t_candidates[0], t_candidates.size()*sizeof(t_candidates[0]));
                    counter += t_candidates.size();
                    //for (auto c : t_candidates) {
                    //    //candidates.push_back(c);
                    //    counter += 1;

                    //}
                }
            }

            auto marker = make_pair(-1, -1);
            out.write((char*)&marker, sizeof(marker));

            //candidates = {};
        }
    }

    dump(counter);
}


template <int HASH_SIZE>
void single_table_loop(
        vector<pair<int, int>> &candidates,
        const int &max_hamming_distance,
        const int &n_tables,
        const int &long_hash_size,
        vector<uint64_t> &vec_n_entries,
        vector<vector<uint64_t>> &vv,
        ofstream &out
    ) {

    int size = vec_n_entries[0];

    uint64_t* v = &vv[0][0];

    uint64_t counter = 0;
    #pragma omp parallel
    {
        vector<pair<int, int>> t_candidates;
        uint64_t a_partial = 0;

        #pragma omp for schedule(static, 256)
        for (int i=0; i<size; i++) {
            uint64_t vi0 = HASH_SIZE*i;
            for (int j=i+1; j<size; j++) {
                uint64_t vj0 = HASH_SIZE*j;
                int dist;
                dist  = __builtin_popcountll(v[vi0  ] ^ v[vj0  ]);
                for(int k=1; k<HASH_SIZE; k++) {
                    dist  += __builtin_popcountll(v[vi0+k] ^ v[vj0+k]);
                }

                if (dist <= max_hamming_distance) {
                    t_candidates.push_back(make_pair(i, j));
                }
            }

        }

        #pragma omp critical
        {
            out.write((char*)&t_candidates[0], t_candidates.size()*sizeof(t_candidates[0]));
            counter += t_candidates.size();
            //for (auto c : t_candidates) {
            //    candidates.push_back(c);
            //}
        }
    }

    dump(counter);
}



int main(int argc, char* argv[]) {
    cout.precision(numeric_limits<double>::max_digits10); // make cout precise
    ios::sync_with_stdio(false); // Disable legacy IO to improve performance

    if (argc != 4) {
        cerr << "Usage: create_canditate_set PATH_TO_INPUT MAX_HAMMING_DISTANCE OUTPUT_PATH" << endl;
        return -1;
    }

    string input_path = argv[1];
    int max_hamming_distance = stoi(argv[2]);
    string output_path = argv[3];

    string extension = get_extension_from_path(input_path);


    // input variables
    int n_tables;
    vector<uint64_t> vec_n_entries = {};
    int hash_size, long_hash_size;
    vector<double> alphas;
    vector<vector<uint64_t>> vv;
    vector<vector<string>> ti_2_i_2_record_id;
    vector<pair<int, int>> candidates;
    vector<vector<int>> ve;
    int max_dist;
    int word_size = 1;


    if (extension != "emb" ) {
        cerr << "error: embedding must be an .emb file" << endl;
        exit(1);
    }

    load_emb(input_path, n_tables, vec_n_entries, hash_size, long_hash_size, vv, ti_2_i_2_record_id);

    dump(hash_size);
    dump(long_hash_size);
    for (int i; i<n_tables; i++) {
        dump(vv[i].size());
        dump(vec_n_entries[i]);
    }


    uint64_t positive = 0;
    uint64_t total = 0;
    dump(hash_size);


    auto start = high_resolution_clock::now();

    ofstream out(output_path, ios::binary);

#define run_multi(N) { \
        multi_table_loop<N>(candidates, max_hamming_distance, n_tables, long_hash_size, \
                vec_n_entries, vv, out); \
    }; \


    if (n_tables > 1) {
        switch(hash_size) {
            case  1: run_multi( 1); break;
            case  2: run_multi( 2); break;
            case  3: run_multi( 3); break;
            case  4: run_multi( 4); break;
            case  5: run_multi( 5); break;
            case  6: run_multi( 6); break;
            case  7: run_multi( 7); break;
            case  8: run_multi( 8); break;
            case  9: run_multi( 9); break;
            case 10: run_multi(10); break;
            case 11: run_multi(11); break;
            case 12: run_multi(12); break;
            case 13: run_multi(13); break;
            case 14: run_multi(14); break;
            case 15: run_multi(15); break;
            case 16: run_multi(16); break;
            case 17: run_multi(17); break;
            case 18: run_multi(18); break;
            case 19: run_multi(19); break;
            case 20: run_multi(20); break;
            case 21: run_multi(21); break;
            case 22: run_multi(22); break;
            case 23: run_multi(23); break;
            case 24: run_multi(24); break;
            case 25: run_multi(25); break;
            case 26: run_multi(26); break;
            case 27: run_multi(27); break;
            case 28: run_multi(28); break;
            case 29: run_multi(29); break;

            default:
                cerr << "error: hash_size=" << long_hash_size << " is too big" << endl;
                exit(1);
                break;
        }


    } else {

#define run_single(N) { \
        single_table_loop<N>(candidates, max_hamming_distance, n_tables, long_hash_size, \
                vec_n_entries, vv, out); \
    }; \

        switch(hash_size) {
            case  1: run_single( 1); break;
            case  2: run_single( 2); break;
            case  3: run_single( 3); break;
            case  4: run_single( 4); break;
            case  5: run_single( 5); break;
            case  6: run_single( 6); break;
            case  7: run_single( 7); break;
            case  8: run_single( 8); break;
            case  9: run_single( 9); break;
            case 10: run_single(10); break;
            case 11: run_single(11); break;
            case 12: run_single(12); break;
            case 13: run_single(13); break;
            case 14: run_single(14); break;
            case 15: run_single(15); break;
            case 16: run_single(16); break;
            case 17: run_single(17); break;
            case 18: run_single(18); break;
            case 19: run_single(19); break;
            case 20: run_single(20); break;
            case 21: run_single(21); break;
            case 22: run_single(22); break;
            case 23: run_single(23); break;
            case 24: run_single(24); break;
            case 25: run_single(25); break;
            case 26: run_single(26); break;
            case 27: run_single(27); break;
            case 28: run_single(28); break;
            case 29: run_single(29); break;
            default:
                cerr << "error: hash_size=" << long_hash_size << " is too big" << endl;
                exit(1);
                break;
        }


    }


    //partial_sum(tp.begin(), tp.end(), tp.begin());
    //partial_sum(fp.begin(), fp.end(), fp.begin());
    //for (int md=0;md<long_hash_size; md++) {
    //    //tp[md] += tp_p[md];
    //    tn[md] = total-positive - fp[md];
    //    //fp[md] += fp_p[md];
    //    fn[md] = positive-tp[md];

    //}

    auto end = high_resolution_clock::now();
    double duration = duration_cast<microseconds>(end - start).count();
    duration = duration / 1e6;


    //for (auto p : candidates) {
    //    cout << p.first << ", " << p.second << endl;
    //}
    //for (int md=0;md<long_hash_size; md++) {
    //    cout << tp[md] << "\t";
    //    cout << tn[md] << "\t";
    //    cout << fp[md] << "\t";
    //    cout << fn[md] << endl;
    //}

    dump(total);
    dump(duration);
    cout << duration << endl;

}
