import argparse
import os
import sys
import json
from time import time

def run(command):
    out, err = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()
    if out:
        out = out.decode('utf-8')
        out = out.strip()
    else:
        out = None

    if err:
        err = err.decode('utf-8')
        err = err.strip()
    else:
        err = None

    return out, err

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help = "path to the tsv/bin file containing the hashed entries."
)

parser.add_argument(
    "-st",
    "--single_thread",
    action="store_true",
    help = "run in single thread instead of multithread."
)

parser.add_argument(
    "-s",
    "--skip_existing",
    action="store_true",
    help = "skip existing evaluations."
)

args = parser.parse_args()


assert (args.path.endswith('.tsv') or args.path.endswith('.bin')) and os.path.exists(args.path), "Invalid file path."

path_splits = args.path.split("/")
hparams_path = "/".join(path_splits[:-1]) + "/hparams.json"

datafold = path_splits[-1].split("_")[0]
out_folder = '/'.join(["research", "eval"]+path_splits[2:-1])
out_path = (out_folder + '/' + datafold + "-full_prediction.json")
os.makedirs(out_folder, exist_ok=True)

if args.skip_existing and os.path.exists(out_path):
    print(f"{out_path} exists, skipping...")
    exit(0)

if not args.single_thread:
    from multiprocessing.dummy import Pool
import subprocess

start = time()
if not args.single_thread:
    out, err = run("lscpu | grep NUMA | grep CPU") #, shell=True, stdout=subprocess.PIPE).communicate()

    list_omp_places = []
    list_omp_threads = []

    for line in out.split('\n'):
        cores = [int(x) for x in line.split()[-1].replace("-", " ").replace(","," ").split()]

        omp_places = ""
        threads = 0
        for i in range(int(len(cores)) // 2):
            omp_places += "{" + str(cores[2*i]) + "}:" + str(cores[2*i+1] - cores[2*i] + 1) +","
            threads += cores[2*i+1] - cores[2*i] + 1

        omp_places = omp_places[:-1]

        list_omp_places.append(omp_places)
        list_omp_threads.append(threads)

    numa_nodes = len(list_omp_places)

    print(f"{numa_nodes} numa nodes:")
    print(' '+'\n '.join(list_omp_places))
    print()
    print(f"threads per node: {list_omp_threads}")
    print()
    sys.stdout.flush()
    commands = []
    for job_id, (omp_places, omp_threads) in enumerate(zip(list_omp_places,list_omp_threads)):
        command = ''
        command += f"OMP_PLACES='{omp_places}' "
        command += f"OMP_NUM_THREADS='{omp_threads}' "
        command += f"OMP_PROC_BIND='close' "
        command += f"bin/eval/hamming {args.path} {numa_nodes} {job_id}"
        commands.append(command)

    p = Pool(numa_nodes)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for out, err in p.imap(run, commands):
        inc = [int(x) for x in out.split('\n')]

        tp += inc[0]
        tn += inc[1]
        fp += inc[2]
        fn += inc[3]

if args.single_thread:
    command = ""
    command += f"OMP_NUM_THREADS='1' "
    command += f"bin/eval/hamming {args.path}"

    out, err = run(command) #, shell=True, stdout=subprocess.PIPE).communicate()
    inc = [int(x) for x in out.split('\n')]

    tp = inc[0]
    tn = inc[1]
    fp = inc[2]
    fn = inc[3]

end = time()

recall = 0 if (tp+fn == 0) else tp / (tp + fn)
precision = 0 if (tp+fp == 0) else tp / (tp + fp)
rr = 1 - (tp + fp) / (tp + fn + fp + tn)



out_dict = {
    "recall": recall,
    "reduction_ratio": rr,
    "precision": precision,
    "h_score": 0 if (rr + recall == 0 ) else 2*(rr * recall)/(rr + recall),
    "f_score": 0 if (precision + recall == 0) else 2*(precision * recall)/(precision + recall),
    "hparams": json.load(open(hparams_path, "r")),
    "output_path": out_path,
    "datafold": datafold,
    "evaluation_time": end-start,
}

print(json.dumps(out_dict, indent=4))

json.dump(out_dict, open(out_path, "w"), indent=4)
