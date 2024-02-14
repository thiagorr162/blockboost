#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

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

template <typename t>
inline int dist(t begin_0, t end_0, t begin_1) {
    int sum = 0;
    for (int i = 0; i< end_0-begin_0; i++) {
        sum += begin_0[i] != begin_1[i];
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
        dataset_path = "research/data/processed/"+dataset+"/test-textual.csv";
    }
    if (basename.find("val") == 0) {
        dataset_path = "research/data/processed/"+dataset+"/val-textual.csv";
    }
    if (basename.find("train") == 0) {
        dataset_path = "research/data/processed/"+dataset+"/train-textual.csv";
    }
    return dataset_path;
}

void load(ifstream &input_file, int &n_tables, vector<uint64_t> &vec_n_entries, int &hash_size, int &max_dist, vector<vector<int>> &vv) {
    // load control variables
    input_file >> n_tables;
    for (int ti=0; ti<n_tables; ti++) {
        int n_entries;
        input_file >> n_entries;
        vec_n_entries.push_back(n_entries);

    }
    input_file >> hash_size;

    input_file >> max_dist;
    // load data
    for (int ti=0; ti<n_tables; ti++) {
        vv.push_back({});

        for (int64_t i=0; i<vec_n_entries[ti]; i++) {
            for (int64_t j=0; j<hash_size+1; j++) {
                int a;
                input_file >> a;
                vv[ti].push_back(a);
            }
        }
    }

    // optimize order of tables, to minimize cache misses (bigger first)
    auto cmp = [](vector<int> a, vector<int> b) {
        return a.size() > b.size();
    };
    sort(vv.begin(), vv.end(), cmp);
    sort(vec_n_entries.begin(), vec_n_entries.end(), [](int a, int b){return a > b;});
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

void load_bin(string &input_path, int &n_tables, vector<uint64_t> &vec_n_entries, int &hash_size, vector<vector<int>> &vv) {
    string shape_path = input_path.substr(0, input_path.length()-3 ) + "shape";
    ifstream shape_file(shape_path, ios::in);
    int total_n_entries;
    int word_size;
    shape_file >> total_n_entries >> hash_size >> word_size;

    ifstream input_file(input_path, ios::binary);
    string dataset_path = get_dataset_path(input_path);

    dump(input_path);
    dump(total_n_entries);
    dump(hash_size);
    dump(word_size);
    dump(dataset_path);

    auto [index_2_ei, index_2_ti] = get_index_2_ei_ti(dataset_path, n_tables);
    int count = 0;

    dump(n_tables);

    // initialize empty tables
    for (int ti=0; ti < n_tables; ti++) {
        vv.push_back({});
    }


    unordered_map<string, int> word_2_int_hash;

    for (int k=0; k<n_tables; k++)
        vec_n_entries.push_back(0);

    for (int i=0; i<total_n_entries; i++) {
        int ei = index_2_ei[i];
        int ti = index_2_ti[i];
        vec_n_entries[ti]++;

        vv[ti].push_back(ei);

        for (int hi=0; hi<hash_size; hi++) {
            string word = "";
            for (int wi=0; wi<word_size; wi++) {
                if (input_file.eof()) {
                    cerr << "error: unexpedted end of file." << endl;
                    dump(input_path);
                    dump(input_file.eof());
                    exit(1);
                }
                int8_t x;
                input_file >> x;
                word += to_string(x)+";";
            }

            dump(word);
            int int_hash=0;
            if (word_2_int_hash.find(word) == word_2_int_hash.end()) {
                word_2_int_hash[word] = word_2_int_hash.size();
            }
            int_hash = word_2_int_hash[word];
            vv[ti].push_back(int_hash);
        }
    }

}

void load_emb(string &input_path, int &n_tables, vector<uint64_t> &vec_n_entries, int &hash_size, vector<vector<int>> &vv) {
    string shape_path = input_path.substr(0, input_path.length()-3 ) + "shape";
    ifstream shape_file(shape_path, ios::in);
    int total_n_entries;
    int word_size;
    shape_file >> total_n_entries;

    ifstream input_file(input_path, ios::binary);
    string dataset_path = get_dataset_path(input_path);

    int64_t input_size = file_size(input_path);

    hash_size = input_size / total_n_entries;

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

    if (argc < 2 || argc > 5) {
        cerr << "usage (job subdivision): " << endl;
        cerr << "  hamming PATH_TO_INPUT N_JOBS JOB_ID" << endl;
        cerr << endl;
        cerr << "usage (job subdivision): " << endl;
        cerr << "  hamming PATH_TO_INPUT N_JOBS JOB_ID MAX_HAMMING_DIST" << endl;
        cerr << endl;
        cerr << "usage (with OMP): " << endl;
        cerr << "  hamming PATH_TO_INPUT" << endl;
        cerr << endl;
        cerr << "usage (with OMP): " << endl;
        cerr << "  hamming PATH_TO_INPUT MAX_HAMMING_DIST" << endl;
        cerr << endl;
        cerr << "the output of the program is:" << endl;
        cerr << "tp" << endl;
        cerr << "tn" << endl;
        cerr << "fp" << endl;
        cerr << "fn" << endl;
        return -1;
    }

    int n_jobs=1, job_id=0;
    string input_path = argv[1];

    string extension = get_extension_from_path(input_path);

    if (argc == 4) {
        n_jobs = stoi(argv[2]);
        job_id = stoi(argv[3]);
        //omp_set_num_threads(1);
    }

    // input variables
    int n_tables;
    vector<uint64_t> vec_n_entries = {};
    int hash_size;
    vector<vector<int>> vv;
    int max_dist;
    int word_size = 1;

    // output variables
    uint64_t tp=0;
    uint64_t tn=0;
    uint64_t fp=0;
    uint64_t fn=0;

    if (extension == "tsv") {

        ifstream input_file(input_path, ios::in);

        if (input_file.fail()) {
            cerr << "error: invalid filepath \"" << input_path << "\"" << endl;
            return 1;
        }
        load(input_file, n_tables, vec_n_entries, hash_size, max_dist, vv);

    } else if (extension == "bin" ) {
        load_bin(input_path, n_tables, vec_n_entries, hash_size, vv);
        max_dist = hash_size - 1;
        dump(input_path);
        dump(n_tables);
        dump(hash_size);
        dump(word_size);
    } else if (extension == "emb" ) {
        load_emb(input_path, n_tables, vec_n_entries, hash_size, vv);

        cerr << "here" << endl;
        if ((argc != 5) && (argc != 3)) {
            cerr << "error: missing max_dist" << endl;
            exit(1);
        }
        istringstream ss(argv[argc-1]);
        if (!(ss >>max_dist)) {
            cerr << "error: invalid max_dist" << argv[4] << endl;
            exit(1);
        }
        dump(max_dist);
    }


    if (n_tables > 1) {

        for (int ti=0; ti<n_tables; ti++) {
            for (int tj=ti+1; tj<n_tables; tj++) {

                #pragma omp parallel
                {
                    uint64_t tp_p=0;
                    uint64_t tn_p=0;
                    uint64_t fp_p=0;
                    uint64_t fn_p=0;

                    uint64_t step = 1 + (vec_n_entries[tj] - 1) / n_jobs;
                    uint64_t j_0 = step * job_id;
                    uint64_t j_f = min(j_0 + step, vec_n_entries[tj]);

                    for (int i=0; i<vec_n_entries[ti]; i++) {
                        auto begin_0 = vv[ti].begin()+(i*(hash_size*word_size+1))+1;
                        auto end_0 = begin_0 + hash_size*word_size;

                        // copying hash variable to a different vector is slightly faster
                        vector<int> v0;
                        for (auto p=begin_0; p<end_0; p++) {
                            v0.push_back(*p);
                        }

                        # pragma omp for
                        for (int j=j_0; j<j_f; j++) {

                            auto begin_1 = vv[tj].begin()+j*(hash_size*word_size+1)+1;

                            //bool is_close = dist(begin_0, end_0, begin_1) <= max_dist;
                            //bool is_close = dist(v0.begin(), v0.end(), begin_1, word_size) <= max_dist;
                            bool is_close = dist(v0.begin(), v0.end(), begin_1) <= max_dist;
                            bool gt = vv[ti][i*(hash_size*word_size+1)] == vv[tj][j*(hash_size*word_size+1)];

                            tp_p += is_close * gt;
                            tn_p += !is_close * !gt;
                            fp_p += is_close * !gt;
                            fn_p += !is_close * gt;
                        }


                    }
                    # pragma omp critical
                    {
                        tp += tp_p;
                        tn += tn_p;
                        fp += fp_p;
                        fn += fn_p;
                    }
                }
            }
        }
    } else {

        #pragma omp parallel
        {
            int ti = 0;
            int tj = 0;
            uint64_t tp_p=0;
            uint64_t tn_p=0;
            uint64_t fp_p=0;
            uint64_t fn_p=0;
            if (!omp_get_thread_num()) {
                dump(tj);
                dump(vec_n_entries.size());
            }
            uint64_t step = 1 + (vec_n_entries[tj] - 1) / n_jobs;
            uint64_t j_0 = step * job_id;
            uint64_t j_f = min(j_0 + step, vec_n_entries[tj]);

            for (uint64_t i=0; i<vec_n_entries[ti]; i++) {
                auto begin_0 = vv[ti].begin()+(i*(hash_size*word_size+1))+1;
                auto end_0 = begin_0 + hash_size*word_size;

                // copying hash variable to a different vector is slightly faster
                vector<int> v0;
                for (auto p=begin_0; p<end_0; p++) {
                    v0.push_back(*p);
                }

                # pragma omp for
                for (uint64_t j=max(j_0, i+1); j<j_f; j++) {

                    auto begin_1 = vv[tj].begin()+j*(hash_size*word_size+1)+1;

                    //bool is_close = dist(begin_0, end_0, begin_1) <= max_dist;
                    //bool is_close = dist(v0.begin(), v0.end(), begin_1, word_size) <= max_dist;
                    bool is_close = dist(v0.begin(), v0.end(), begin_1) <= max_dist;
                    bool gt = vv[ti][i*(hash_size*word_size+1)] == vv[tj][j*(hash_size*word_size+1)];

                    tp_p += is_close * gt;
                    tn_p += !is_close * !gt;
                    fp_p += is_close * !gt;
                    fn_p += !is_close * gt;
                }


            }
            # pragma omp critical
            {
                tp += tp_p;
                tn += tn_p;
                fp += fp_p;
                fn += fn_p;
            }
        }

    }

    cout << tp << endl;
    cout << tn << endl;
    cout << fp << endl;
    cout << fn << endl;
    //if (n_jobs == 1) {
    //    cout << "total_pairs=" << total_pairs << endl;
    //    cout << endl;
    //    double recall = tp / ((double) tp + (double) fn);
    //    double precision = tp / ((double) tp + (double) fp);
    //    double reduction_ratio = 1.0 - (tp+fp) / total_pairs;
    //    double h;
    //    if (recall + reduction_ratio > 0)
    //        h = 2 * recall * reduction_ratio / (recall + reduction_ratio);
    //    else
    //        h = -1;

    //    cout << "recall=" << recall << endl;
    //    cout << "precision=" << precision << endl;
    //    cout << "reduction_ratio=" << reduction_ratio << endl;
    //    cout << "h=" << h << endl;
    //}
}
