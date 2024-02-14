#include <bits/stdc++.h>
#include <boost/sort/sort.hpp>

using namespace boost::sort;
using namespace std::chrono;
using namespace std;

#define IGNORE_RETURN(x) if (x) {}

#define doubledouble 1
#define floatfloat 1

#define concat_literal(x,y) x ## y
#define realtype(t1,t2) concat_literal(t1,t2)

//#define REAL float

#define DEFAULT_NAN -9999
#define DEFAULT_STR_NAN "-9999"

#define EPS 1e-14
#define PROCESSED_PATH "research/data/processed/"
#define DEBUG
//#define VDEBUG

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

//#define SAVE_TRAIN 1 to save train

vector<REAL> A;
vector<REAL> B;
vector<int8_t> y;
uint64_t num_pos;
uint64_t n;
uint64_t d;

vector<REAL> w;
vector<vector<REAL>> id_2_thresholds;

vector<tuple<REAL, int>> max_w;
vector<tuple<REAL, int>> min_w;


double load_time;
double preprocess_time;
double training_time;
double prediction_time;

template <typename T>
int sign(const T &a) {
    return (a + EPS >= 0) - (a + EPS < 0);
}

template <class T>
vector<T> transpose(vector<T> &M, uint64_t nl, uint64_t nc) {
    vector<T> Mt(M.size());

    for (uint64_t i=0; i<nl; i++) {
        for (uint64_t j=0; j<nc; j++) {
            Mt[j*nl + i] =  M[i*nc + j];
        }
    }

    return Mt;
}


template <class T>
void load_thresholds(vector<T> &thresholds, uint64_t i, const vector<T> &A, const vector<T> &B) {
    thresholds = {};
    thresholds.reserve(n);
    //vector<T> thresholds = {};
    for (uint64_t j=0; j<n; j++) {
        thresholds.push_back(A[j + i*n]);
        thresholds.push_back(B[j + i*n]);
    }
    block_indirect_sort(thresholds.begin(), thresholds.end());
    //sort(thresholds.begin(), thresholds.end());
    thresholds.erase(
            unique(thresholds.begin(), thresholds.end()),
            thresholds.end());
    //thresholds = vector<T>(set_thresholds.begin(), set_thresholds.end());
}

void load_dataset(string path, int seed, int pm, int pn) {
    double total_time = 0;
    cerr << "loading file..." << endl;
    auto start = high_resolution_clock::now();
    ifstream f(path);

    if (!f) {
        cerr << "error reading '" << path << "'" << endl;
        exit(1);
    }

    unordered_map<string, int> seid_2_eid;

    vector<pair<int, int>> v_eid_rid;

    vector<REAL> emb;

    int n_emb = 0;
    d = -1;

    // discard header
    string line;
    getline(f, line);

    for (int i=0; i<line.size(); i++)
        d += line[i] == ',';

    // load file
    while(!f.eof()) {
        string line;
        getline(f, line);

        if (line.size() > 2) {
            stringstream ss(line);
            vector<string> cells;

            while (ss.good()) {
                string cell;
                getline(ss, cell, ',');
                if (cell != "")
                    cells.push_back(cell);
                else
                    cells.push_back(DEFAULT_STR_NAN);
            }
            if (cells.size()-2 != d){
                cerr << "error on line " << n_emb+1 <<", number of dimensions ("<< cells.size()<< ") is different from header (" << d << ")" << endl;
                exit(1);
            }

            for (int i=0; i<cells.size()-2; i++) {
                emb.push_back(stod(cells[i]));
            }

            string seid = cells[cells.size()-2];
            string srid = cells[cells.size()-1];

            int eid;
            int rid = n_emb;

            if (seid_2_eid.find(seid) == seid_2_eid.end()) {
                eid = seid_2_eid.size();
                seid_2_eid[seid] = eid;
            } else {
                eid = seid_2_eid[seid];
            }

            v_eid_rid.push_back(make_pair(eid, rid));

            n_emb++;
        }
    }
    f.close();
    dump(n_emb);
    dump(d);

    auto end = high_resolution_clock::now();
    double duration = pow(10,-3)* (double)duration_cast<milliseconds>(end - start).count();
    total_time += duration;
    cerr << "loaded file in " << duration << "s" << endl << endl;

    cerr << "constructing pairs..." << endl;
    start = high_resolution_clock::now();

    vector<pair<int, int>> positive_pairs;
    vector<pair<int, int>> negative_pairs;

    // sort struct to group by entity_id
    block_indirect_sort(v_eid_rid.begin(), v_eid_rid.end());


    // populate positive_pairs
    auto it = v_eid_rid.begin();
    while(it+1 != v_eid_rid.end()) {
        if (it->first == (it+1)->first) {
            auto begin_pos = it;

            while (it+1 != v_eid_rid.end() && (it+1)->first == it->first) {
                it++;
            }
            auto end_pos = it+1;

            for (auto it_pos1 = begin_pos; it_pos1 < end_pos; it_pos1++) {
                for (auto it_pos2 =begin_pos; it_pos2 < end_pos; it_pos2++) {
                    if (it_pos1 != it_pos2) {
                        for (int i=0; i<pm; i++)
                            positive_pairs.push_back(make_pair(it_pos1->second, it_pos2->second));
                        vdump(it_pos1->first);
                        vdump(it_pos1->second);
                        vdump(it_pos2->first);
                        vdump(it_pos2->second);
                        vdump("");
                    }
                }
            }
        } else {
            it++;
        }
    }

    num_pos = positive_pairs.size();

    // populate negative_pairs
    mt19937 rng;
    rng.seed(seed);
    uniform_int_distribution<int> rnd_index(0,n_emb-1);
    while (negative_pairs.size() < positive_pairs.size()*pn) {
        int i1, i2;
        do {
            i1 = rnd_index(rng);
            i2 = rnd_index(rng);
        } while (v_eid_rid[i1].first == v_eid_rid[i2].first);

        negative_pairs.push_back(make_pair(i1, i2));
    }


    end = high_resolution_clock::now();
    duration = pow(10,-3)* (double)duration_cast<microseconds>(end - start).count();
    dump(positive_pairs.size());
    dump(negative_pairs.size());
    total_time += duration * pow(10,-3);
    cerr << "constructed pairs in " << duration << "ms" << endl << endl;


    cerr << "constructing A, B and y..." << endl;
    start = high_resolution_clock::now();

    // construct matrices A and B
    for(int i=0; i<positive_pairs.size(); i++) {
        int ri_A = positive_pairs[i].first;
        int ri_B = positive_pairs[i].second;
        for (int j=0; j<d; j++) {
            A.push_back(emb[ri_A*d + j]);
        }
        for (int j=0; j<d; j++) {
            B.push_back(emb[ri_B*d + j]);
        }
    }

    for(int i=0; i<negative_pairs.size(); i++) {
        int ri_A = negative_pairs[i].first;
        int ri_B = negative_pairs[i].second;
        for (int j=0; j<d; j++) {
            A.push_back(emb[ri_A*d + j]);
        }
        for (int j=0; j<d; j++) {
            B.push_back(emb[ri_B*d + j]);
        }
    }

    n = positive_pairs.size() + negative_pairs.size();
    A = transpose(A, n, d);
    B = transpose(B, n, d);

    for (int i=0; i<positive_pairs.size(); i++) {
        y.push_back(1);
    }
    for (int i=0; i<negative_pairs.size(); i++) {
        y.push_back(-1);
    }

    dump(y.size());

    end = high_resolution_clock::now();
    duration = pow(10,-3)* (double)duration_cast<microseconds>(end - start).count();
    cerr << "constructed A, B and y in " << duration << "ms" << endl << endl;
    total_time += duration * pow(10,-3);
    cerr << "total time " << total_time << "s" << endl << endl;
    load_time = total_time;
}

void get_weak_learner(int *bf, double *bt, double *be) {

    vdump(n);
    vdump(d);

    int slow_best_feature = 0;
    double slow_best_threshold = 0.0;
    double slow_best_error = 1.0;

    int fast_best_feature = 0;
    double fast_best_threshold = 0.0;
    double fast_best_error = 1.0;

    long unsigned int max_num_thresholds = 0;
    for (int i=0; i<d; i++)
        max_num_thresholds = max(id_2_thresholds[i].size(), max_num_thresholds);

    #pragma omp parallel
    {
        int local_best_feature = 0;
        double local_best_threshold = 0.0;
        double local_best_error = 1.0;

        vector<tuple<REAL, REAL>> local_max;
        vector<tuple<REAL, REAL>> local_min;
        local_max.reserve(max(num_pos, y.size()-num_pos));
        local_min.reserve(max(num_pos, y.size()-num_pos));
        vector<REAL> ti_to_w;
        ti_to_w.reserve(max_num_thresholds);

        vector<REAL> _ti_to_w;
        _ti_to_w.reserve(max_num_thresholds);

        #pragma omp for schedule(dynamic)
        for (uint64_t i=0; i<d; i++) {

            const vector<REAL> vt = id_2_thresholds[i];
            ti_to_w.resize(vt.size());
            fill(ti_to_w.begin(), ti_to_w.end(), 0);

            for (int y_sign=-1; y_sign<2; y_sign+=2) {

                //vector<double> _ti_to_w(vt.size());
                _ti_to_w.resize(vt.size());
                fill(_ti_to_w.begin(), _ti_to_w.end(), 0);

                if (y_sign == 1) {
                    local_max.resize(num_pos);
                    local_min.resize(num_pos);
                } else {
                    local_max.resize(y.size()-num_pos);
                    local_min.resize(y.size()-num_pos);
                }

                int max_i = 0;
                int min_i = 0;
                for (uint64_t j=0;j<n; j++) {
                    uint64_t ind = i*n + j;
                    auto item_max = max_w[ind];

                    if (y[get<1>(item_max)] == y_sign) {
                        //local_max.push_back(
                        //    make_tuple(get<0>(item_max),
                        //               w[get<1>(item_max)]));
                        //
                        local_max[max_i] =
                            make_tuple(get<0>(item_max),
                                       w[get<1>(item_max)]);
                        max_i++;
                    }

                    auto item_min = min_w[ind];
                    if (y[get<1>(item_min)] == y_sign) {
                        //local_min.push_back(
                        //    make_tuple(get<0>(item_min),
                        //               w[get<1>(item_min)]));
                        local_min[min_i] =
                            make_tuple(get<0>(item_min),
                                       w[get<1>(item_min)]);
                        min_i++;
                    }
                }


                //double wp += (A[k]<t && B[k]<t) * w[k];
                for (auto it=local_max.begin()+1; it!=local_max.end(); it++) {
                    get<1>(*it) += get<1>(*(it-1));
                }

                //double wp += (A[k]>t && B[k]>t) * w[k];
                for (auto it=local_min.rbegin()+1; it!=local_min.rend(); it++) {
                    get<1>(*it) += get<1>(*(it-1));
                }


                //double wp += (A[k]<t && B[k]<t) * w[k];
                auto it_max = local_max.begin();
                REAL db_local_max = 0;
                for (uint64_t ti=0; ti<_ti_to_w.size(); ti++) {
                    while((get<0>(*it_max) < vt[ti]) && it_max!=local_max.end()) {
                        //dump(db_local_max);
                        db_local_max = get<1>(*it_max);
                        //dump(db_local_max);
                        //dump("");
                        it_max++;
                    }
                    _ti_to_w[ti] += db_local_max;
                }

                //double wp += (A[k]>t && B[k]>t) * w[k];
                auto it_min = local_min.rbegin();
                db_local_max = 0;
                for (int ti=ti_to_w.size()-1; ti>=0; ti--) {
                    while((get<0>(*it_min) >= vt[ti]) && it_min != local_min.rend()) {
                        //dump(db_local_max);
                        db_local_max = get<1>(*it_min);
                        //dump(db_local_max);
                        //dump("");
                        it_min++;
                    }
                    _ti_to_w[ti] += db_local_max;
                }

                if (y_sign > 0) {
                    auto mw = max_element(_ti_to_w.begin(), _ti_to_w.begin());
                    for (int ti=0; ti<ti_to_w.size(); ti++) {
                        ti_to_w[ti] += *mw - _ti_to_w[ti];
                    }
                } else {
                    for (int ti=0; ti<ti_to_w.size(); ti++) {
                        ti_to_w[ti] += _ti_to_w[ti];
                    }
                }
            }
            //for (double w : ti_to_w) {
            //    dump(w);
            //}

            auto it_min_err = min_element(ti_to_w.begin(), ti_to_w.end());

            if (*it_min_err < local_best_error) {
                local_best_error = *it_min_err;
                local_best_feature = i;
                local_best_threshold = vt[it_min_err-ti_to_w.begin()];
            }
        }

#pragma omp critical
        if (local_best_error < fast_best_error) {
            fast_best_error = local_best_error;
            fast_best_feature = local_best_feature;
            fast_best_threshold = local_best_threshold;
        }

    }

    vdump(fast_best_feature);
    vdump(fast_best_threshold);
    vdump(fast_best_error);

    *bf = fast_best_feature;
    *bt = fast_best_threshold;
    *be = fast_best_error;
}

void init() {
    cerr << "initializing auxiliary structs O(d n (log n))..." << endl;

    dump(n);
    dump(d);
    auto start = high_resolution_clock::now();

    for (uint64_t i=0; i<d; i++) {
        vector<REAL> thresholds;
        load_thresholds(thresholds, i, A, B);
        id_2_thresholds.push_back(thresholds);
    }
    auto end_1 = high_resolution_clock::now();
    double duration_1 = pow(10,-3)* (double)duration_cast<milliseconds>(end_1 - start).count();
    cerr << "thresholds " << duration_1 << "s" << endl << endl;

    max_w.reserve(n*d);
    min_w.reserve(n*d);

    for (uint64_t i=0; i<d; i++) {
        for (uint64_t j=0;j<n; j++) {
            max_w.push_back(make_tuple(max(A[j + i*n], B[j + i*n]), j));
            min_w.push_back(make_tuple(min(A[j + i*n], B[j + i*n]), j));
        }

        block_indirect_sort(max_w.begin()+i*n,
             max_w.begin()+(i+1)*n,
             [](auto a, auto b){ return get<0>(a) < get<0>(b);});

        block_indirect_sort(min_w.begin()+i*n,
             min_w.begin()+(i+1)*n,
             [](auto a, auto b){ return get<0>(a) < get<0>(b);});

    }
    auto end_2 = high_resolution_clock::now();
    double duration_2 = pow(10,-3)* (double)duration_cast<milliseconds>(end_2 - end_1).count();
    cerr << "sorting " << duration_2 << "s" << endl << endl;


    double initial_weight = 1.0 / (double) n;

    for(uint64_t i=0; i<n; i++) {
        w.push_back(initial_weight);
    }

    auto end = high_resolution_clock::now();
    double duration = pow(10,-3)* (double)duration_cast<milliseconds>(end - start).count();
    cerr << "initialized in " << duration << "s" << endl << endl;
    preprocess_time = duration;
}


int main(int argc, char *argv[]) {
    cout.precision(numeric_limits<double>::max_digits10); // make cout precise
    ios::sync_with_stdio(false); // Disable legacy IO to improve performance

    if (argc != 6) {
        cerr << "Usage: train DATASET SEED PM PN N_ITER" << endl;
        cerr << endl;
        cerr << "  DATASET       name of the pre-processed dataset" << endl;
        cerr << "  SEED          seed for stump generation" << endl;
        cerr << "  PM            proportion of matches" << endl;
        cerr << "  PN            proportion of non-matches" << endl;
        cerr << "  N_ITER        maximum number of iterations" << endl;
        exit(1);
    }

    string dataset = argv[1];
    int seed = stoi(argv[2]);
    int pm = stoi(argv[3]);
    int pn = stoi(argv[4]);
    int n_iter = stoi(argv[5]);

    string path = PROCESSED_PATH + dataset + "/train-vectorized.csv";
#ifdef SAVE_TRAIN
    string output_dir = "research/models/blockboost-fp32-save_train/" + dataset + "/s_" +to_string(seed) +"-pm_" + to_string(pm) +"-pn_" + to_string(pn) + "-n_" + to_string(n_iter) + "/embedding";
#else
#if realtype(REAL, double)
    string output_dir = "research/models/blockboost/" + dataset + "/s_" +to_string(seed) +"-pm_" + to_string(pm) +"-pn_" + to_string(pn) + "-n_" + to_string(n_iter) + "/embedding";
#elif realtype(REAL, float)
    string output_dir = "research/models/blockboost-fp32/" + dataset + "/s_" +to_string(seed) +"-pm_" + to_string(pm) +"-pn_" + to_string(pn) + "-n_" + to_string(n_iter) + "/embedding";
#endif
#endif
    string cmd_output_dir = "mkdir -p '"+output_dir+"'";
    if (system(cmd_output_dir.c_str())) {
        cerr << "error: failed to create output dir " << output_dir << endl;
        exit(1);
    }

    load_dataset(path, seed, pm, pn);
    init();

    vector<pair<int, REAL>> history;
    vector<int> bf_history;
    vector<double> alpha_history;
    vector<double> error_history;
    vector<double> normalizer_history;

    auto start = high_resolution_clock::now();
    for(uint64_t i=0; i<n_iter; i++) {
        int bf;
        double bt, be;
        get_weak_learner(&bf, &bt, &be);

        // avoid edge cases
        be = max(be, EPS);
        be = min(be, 1-EPS);
        double alpha = log((1.0 - be) / be) * .5;

        auto now = high_resolution_clock::now();
        double duration = pow(10,-6)* (double)duration_cast<microseconds>(now - start).count();
        duration =max(duration, EPS);
        if (alpha - EPS < 0) {
            cerr  <<setw(7) <<i+1 << ": alpha=" << fixed << setprecision(7) << alpha << "<0, best_error="<<setprecision(8) << be << ", " << (i+1)/duration<< " it/s" << endl;
            break;
        } else {
            cerr <<setw(7)  << i+1 << ": alpha=" << fixed << setprecision(8) << alpha << ">0, best_error="<< be << ", " << (i+1)/duration<< " it/s" << endl;
            history.push_back(make_pair(bf, bt));
            alpha_history.push_back(alpha);
            error_history.push_back(be);
            bf_history.push_back(bf);
        }

        // compute normalizer constant
        double normalizer = 2 * sqrt(be * (1 - be));
        normalizer_history.push_back(normalizer);

        for (uint64_t j=0; j<n; j++) {
            int hashed_A = sign(A[bf*n + j]  - bt);
            int hashed_B = sign(B[bf*n + j]  - bt);

            double exponent = alpha * y[j] * hashed_A * hashed_B;

            w[j] = exp(-exponent) * w[j] / normalizer;
        }

        double acc_w = accumulate(w.begin(), w.end(), 0.0);
        if (abs(acc_w - 1) > 1e-2) {
            cerr << "acc_w does not add to 1" << endl;
            dump(acc_w);
            exit(1);
        }
        vdump(acc_w);
    }
    auto end = high_resolution_clock::now();
    training_time = pow(10,-3)* (double)duration_cast<milliseconds>(end - start).count();

    cerr << endl;
    cerr << "saving preditions..." << endl;
    vector<string> folds = {"val", "test"};

#ifdef SAVE_TRAIN
    folds.push_back("train");
#endif

    start = high_resolution_clock::now();
    for (string fold : folds) {
        string alphas_path = output_dir + "/"+fold+"_prediction.alphas";
        ofstream alphas_file(alphas_path, ios::binary);
        if (!alphas_file.good()) {
            cerr << "error writing to '" << alphas_path << "'" << endl;
        }
        alphas_file.write((char*) &alpha_history[0], alpha_history.size()*sizeof(double));

        alphas_file.close();

        string bf_history_path = output_dir + "/" + fold + "_prediction.features";
        ofstream bf_history_file(bf_history_path);

        if (!bf_history_file) {
            cerr << "error opening " << bf_history_path << " for writing" << endl;
            exit(1);
        }

        double acc_alpha = accumulate(alpha_history.begin(), alpha_history.end(), 0.0);
        bf_history_file << "[" << endl;

        for (int i=0; i < bf_history.size()-1; i++) {
            bf_history_file << "  [" << bf_history[i] << ", " << alpha_history[i] /acc_alpha << "],\n";
        }
        bf_history_file << "  [" << bf_history[bf_history.size()-1] << ", " << alpha_history[bf_history.size()-1] / acc_alpha << "]\n]\n";

        string emb_path = output_dir + "/" + fold + "_prediction.emb";
        ofstream emb_file(emb_path, ios::binary);
        if (!emb_file) {
            cerr << "error opening " << emb_path << " for writing" <<endl;
            exit(1);
        }

        string data_path = PROCESSED_PATH + dataset + "/"+ fold + "-vectorized.csv";
        //ifstream data_file(data_path);

        //if (!data_file) {
        //    cerr << "error opening " << data_path << " for reading" << endl;
        //    exit(1);
        //}
        //// discard header
        //string line;
        //getline(data_file, line);
        FILE *fp;
        fp = fopen(data_path.c_str(), "r");
        char line_str[1024*1024];
        // discard header
        IGNORE_RETURN(fgets(line_str, 1024*1024, fp));



        uint64_t n_pred = 0;
        vector<int8_t> emb_vec;
        while (fgets(line_str, 1024*1024, fp)) {
            const char sep[2] = ",";
            const char* cell = strtok(line_str, sep);
            vector<REAL> vd;
            while( cell != NULL) {
                //printf(" %s\n", cell);
                REAL db_cell = DEFAULT_NAN;

                if (cell[0] != '\0')
                    db_cell = atof(cell);
                vd.push_back(db_cell);

                cell = strtok(NULL,sep);
            }
            for (auto h : history) {
                int8_t x = sign(vd[h.first] - h.second);
                emb_vec.push_back(x);
            }
            n_pred++;

            //getline(data_file, line);
            //vdump(line);
            //if (line.size() > 2) {
            //    vector<REAL> vd;
            //    stringstream ss(line);
            //    for (int j=0; j<d; j++) {
            //        string cell;
            //        getline(ss, cell, ',');

            //        REAL db_cell = DEFAULT_NAN;
            //        if (cell.size() >= 1)
            //            db_cell = stod(cell);
            //        vdump(cell);
            //        vdump(db_cell);
            //        vd.push_back(db_cell);
            //    }
            //    vector<int8_t> emb_vec;
            //    for (auto h : history) {
            //        int8_t x = sign(vd[h.first] - h.second);
            //        emb_vec.push_back(x);
            //    }
            //    emb_file.write((char*)&emb_vec[0], emb_vec.size());
            //    n_pred++;
            //}
        }
        emb_file.write((char*)&emb_vec[0], emb_vec.size());
        emb_file.close();
        fclose(fp);

        string shape_path = output_dir + "/" + fold + "_prediction.shape";
        ofstream shape_file(shape_path);

        if (!shape_file) {
            cerr << "error opening " << shape_path << " for writing" << endl;
            exit(1);
        }
        shape_file << n_pred << "\t" << alpha_history.size() << "\t" << 1 << endl;
        shape_file.close();

    }
    end = high_resolution_clock::now();
    prediction_time = pow(10,-6)* (double)duration_cast<microseconds>(end - start).count();

    string hparams_path = output_dir + "/hparams.json";
    ofstream hparams_file(hparams_path);

    if (!hparams_file) {
        cerr << "error opening " << hparams_path << " for writing" << endl;
        exit(1);
    }

    hparams_file << "{" << endl;
    hparams_file << R"(  "database": ")" +dataset + "\"," << endl;
    hparams_file << R"(  "proportion_matches": )" +to_string(pm) + "," << endl;
    hparams_file << R"(  "proportion_nonmatches": )" + to_string(pn)+"," << endl;
    hparams_file << R"(  "n_iter": )"+ to_string(n_iter)+"," << endl;
    hparams_file << R"(  "seed": )"+ to_string(seed)+"," << endl;
    hparams_file << R"(  "model": "blockboost")" << endl;
    hparams_file << "}" << endl;

    hparams_file.close();

    string info_path = output_dir + "/info.json";
    ofstream info_file(info_path);

    ostringstream info_ss;

    info_ss << "{" << endl;
    info_ss << R"(  "database": ")" +dataset + "\"," << endl;
    info_ss << R"(  "n_train": )" +to_string(n) + "," << endl;
    info_ss << R"(  "n_bits": )" << alpha_history.size() << "," << endl;
    info_ss << R"(  "seed": )"+ to_string(seed)+"," << endl;
    info_ss << R"(  "model": "blockboost",)" << endl;
    info_ss << R"(  "load_time": )" << load_time << "," << endl;
    info_ss << R"(  "preprocess_time": )" << preprocess_time << "," << endl;
    info_ss << R"(  "training_time": )" << training_time << "," << endl;
    info_ss << R"(  "training_time_per_iteration": )" << training_time / alpha_history.size() << "," << endl;
    info_ss << R"(  "prediction_time": )" << prediction_time << ","  << endl;
    info_ss << R"(  "output_dir": ")" << output_dir << "\""  << endl;
    info_ss << "}" << endl;

    cerr << info_ss.str();
    info_file << info_ss.str();

    info_file.close();
}
