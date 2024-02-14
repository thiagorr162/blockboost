#! /bin/octave -qf

addpath('src/models/klsh_octave/');

if nargin != 1
    printf("usage:\n");
    printf("<program> <dataset_name>\n\n");
    exit();
end

arg_list = argv();

dataset_name = arg_list{1};

train_mtx_path = strcat("research/data/processed-klsh_octave/",dataset_name,"/train.mtx");
train_ids_path = strcat("research/data/processed-klsh_octave/",dataset_name,"/train.ids");


%demo script for KLSH
X = load(train_mtx_path);
y = load(train_ids_path);
y = y + 1;
[n,d] = size(X);

%form RBF over the data:
nms = sum(X'.^2);
K = exp(-nms'*ones(1,n) -ones(n,1)*nms + 2*X*X');

size(nms)
size(X)
size(K)

%split into training (database) and test (queries)
rp = randperm(n);
trI = rp(1:ceil(n/2));
testI = rp(ceil(n/2)+1:end);
Ktrain = K(trI,trI);
Ktest = K(testI,trI);
ytrain = y(trI);
ytest = y(testI);

%form KLSH hash table
[H W] = createHashTable(Ktrain,300,30);
H_query = (Ktest*W)>0;

%form distance matrix between queries and database items
CH = compactbit(H);
CH_query = compactbit(H_query);
Dist = hammingDist(CH_query,CH);

%look at a few results
disp('Displaying class labels of some queries (first value in each row),');
disp('along with labels of the first 5 nearest hashed neighbors');
for i = 1:5
    [v, ind] = sort(Dist(i,:),'ascend');
    [ytest(i) ytrain(ind(1:5))']
end
