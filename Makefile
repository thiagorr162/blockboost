all: bin/number_of_matches_of_subsets bin/number_of_matches_of_subsets-over-comb bin/eval/hamming bin/eval/embedding bin/eval/knn bin/eval/make_example  bin/models/blockboost/train bin/data/eval/similarity_measure  bin/models/blockboost/train-omp bin/benchmark/blocking bin/benchmark/predict  bin/benchmark/atof bin/eval/fast_emb bin/models/blockboost/create_candidate_set

bin/number_of_matches_of_subsets: src/data/eval/number_of_matches_of_subsets.cpp
	mkdir -p bin
	g++ -std=c++17 -O3 src/data/eval/number_of_matches_of_subsets.cpp -o bin/number_of_matches_of_subsets

bin/number_of_matches_of_subsets-over-comb: src/data/eval/number_of_matches_of_subsets-over-comb.cpp
	mkdir -p bin
	g++ -std=c++17 -O3 src/data/eval/number_of_matches_of_subsets-over-comb.cpp -o bin/number_of_matches_of_subsets-over-comb

bin/eval/hamming: src/eval/hamming.cpp
	mkdir -p bin/eval
	g++ -std=c++17 -march=native -fopenmp -O3 src/eval/hamming.cpp -o bin/eval/hamming

bin/eval/embedding: src/eval/embedding.cpp
	mkdir -p bin/eval
	g++ -std=c++17  -march=native -fopenmp -O3 src/eval/embedding.cpp -o bin/eval/embedding

bin/eval/knn: src/eval/knn.cpp
	mkdir -p bin/eval
	g++ -std=c++17 -march=native -O3 src/eval/knn.cpp -o bin/eval/knn

bin/eval/make_example: src/eval/make_example.cpp
	mkdir -p bin/eval
	g++ -std=c++17 -march=native -fopenmp -O3 src/eval/make_example.cpp -o bin/eval/make_example


bin/models/blockboost/train bin/models/blockboost/train-omp: src/models/blockboost/train.cpp
	mkdir -p bin/models/blockboost
	mkdir -p bin/models/blockboost-fp32
	mkdir -p bin/models/blockboost-fp32-save_train

	#g++ -std=c++17 -march=native -fopenmp -g3 -O3 src/models/blockboost/train.cpp -o bin/models/blockboost/train-omp
	g++ -std=c++17 -march=native -DREAL=double -fopenmp -O3 src/models/blockboost/train.cpp -o bin/models/blockboost/train-omp
	g++ -std=c++17 -march=native -DREAL=double -O3 src/models/blockboost/train.cpp -o bin/models/blockboost/train

	g++ -std=c++17 -march=native -DREAL=float -fopenmp -O3 src/models/blockboost/train.cpp -o bin/models/blockboost-fp32/train-omp
	g++ -std=c++17 -march=native -DREAL=float -O3 src/models/blockboost/train.cpp -o bin/models/blockboost-fp32/train

	g++ -std=c++17 -march=native -DSAVE_TRAIN=1 -DREAL=float -fopenmp -O3 src/models/blockboost/train.cpp -o bin/models/blockboost-fp32-save_train/train-omp
	g++ -std=c++17 -march=native -DSAVE_TRAIN=1 -DREAL=float -O3 src/models/blockboost/train.cpp -o bin/models/blockboost-fp32-save_train/train


bin/models/blockboost/create_candidate_set: src/models/blockboost/create_candidate_set.cpp
	g++ -std=c++17 -fopenmp -march=native -DREAL=float -O3 src/models/blockboost/create_candidate_set.cpp -o bin/models/blockboost-fp32/create_candidate_set
	cp bin/models/blockboost-fp32/create_candidate_set bin/models/blockboost/create_candidate_set


bin/data/eval/similarity_measure: src/data/eval/similarity_measure.cpp
	mkdir -p bin/data/eval/
	g++ -std=c++17 -march=native -pedantic -O3 src/data/eval/similarity_measure.cpp -o bin/data/eval/similarity_measure

bin/eval/fast_emb: src/eval/fast_embedding.cpp
	mkdir -p bin/eval
	g++ -std=c++17  -march=native -fopenmp -O3 src/eval/fast_embedding.cpp -o bin/eval/fast_emb

bin/benchmark/blocking: src/paper/benchmark_blocking.cpp
	mkdir -p bin/benchmark
	g++ -std=c++17  -march=native -fopenmp -O3 src/paper/benchmark_blocking.cpp -o bin/benchmark/blocking

bin/benchmark/predict: src/paper/benchmark_predict.cpp
	mkdir -p bin/benchmark
	g++ -std=c++17  -march=native -fopenmp -O3 src/paper/benchmark_predict.cpp -o bin/benchmark/predict

bin/benchmark/atof: src/paper/benchmark_atof.cpp
	mkdir -p bin/benchmark
	g++ -std=c++17  -march=native -fopenmp -O3 src/paper/benchmark_atof.cpp -o bin/benchmark/atof
