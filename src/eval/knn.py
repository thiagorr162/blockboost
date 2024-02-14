import argparse
import os
import sys
import json
from time import time
import subprocess

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
    help = "path to the .emb file containing the hashed entries."
)

parser.add_argument(
    "-o",
    "--stdout",
    action="store_true",
    help = "write results in stdout instead of file"
)

#parser.add_argument(
#    "-s",
#    "--skip_existing",
#    action="store_true",
#    help = "skip existing evaluations."
#)

args = parser.parse_args()


assert (args.path.endswith('.emb')) and os.path.exists(args.path), "Invalid file path."

path_splits = args.path.split("/")
hparams_path = "/".join(path_splits[:-1]) + "/hparams.json"

datafold = path_splits[-1].split("_")[0]
out_folder = '/'.join(["research", "eval"]+path_splits[2:-1]+["knn"])
out_folder_ponderated = '/'.join(["research", "eval"]+path_splits[2:-1]+["ponderated_knn"])
os.makedirs(out_folder, exist_ok=True)
os.makedirs(out_folder_ponderated, exist_ok=True)



start = time()

command = ""
command += f"OMP_NUM_THREADS='1' "
command += f"bin/eval/knn {args.path}"

out, err = run(command) #, shell=True, stdout=subprocess.PIPE).communicate()
lines = [x for x in out.strip().split('\n')]

for md, line in enumerate(lines):


    inc = [int(x) for x in line.split('\t')]
    tp = inc[0]
    tn = inc[1]
    fp = inc[2]
    fn = inc[3]

    end = time()

    recall = 0 if (tp+fn == 0) else tp / (tp + fn)
    precision = 0 if (tp+fp == 0) else tp / (tp + fp)
    rr = 1 - (tp + fp) / (tp + fn + fp + tn)

    hparams = json.load(open(hparams_path, "r"))


    if md < len(lines) / 2:
        out_path = (out_folder + '/' + datafold + f"-k_{md}.json")
        hparams["k"] = md
        hparams["metric_type"] = "hamming"
    else:
        out_path = (out_folder_ponderated + '/' + datafold + f"-ponderated-k_{md - len(lines)/2}.json")
        hparams["k"] = md - len(lines) / 2
        hparams["metric_type"] = "ponderated hamming"


    if "bucket_size" in hparams:
        hparams.pop("bucket_size")
        hparams.pop("n_buckets")

    out_dict = {
        "recall": recall,
        "reduction_ratio": rr,
        "precision": precision,
        "h_score": 0 if (rr + recall == 0 ) else 2*(rr * recall)/(rr + recall),
        "f_score": 0 if (precision + recall == 0) else 2*(precision * recall)/(precision + recall),
        "hparams": hparams,
        "output_path": out_path,
        "datafold": datafold,
        "evaluation_time": end-start,
    }

    #print(json.dumps(out_dict, indent=4))

    if not args.stdout:
        json.dump(out_dict, open(out_path, "w"), indent=4)
    else:
        print(json.dumps(out_dict))
