'''

pip install datasets pyarrow

Examples:

# Local files (train & test)
python parquet_to_jsonl.py \
  --parquet train:~/datasets/sft_data/train.parquet test:~/datasets/sft_data/test.parquet \
  --text-column messages \
  --out-dir ~/datasets/megatron_sft_data

# Single parquet (treated as train)
python parquet_to_jsonl.py \
  --parquet ~/datasets/sft_data/only_train.parquet \
  --text-column messages \
  --out-dir  ~/datasets/megatron_sft_data


This produces:

~/datasets/megatron_sft_data/train.jsonl
~/datasets/megatron_sft_data/test.jsonl


(each line like {"text": "…document…"} )


'''

# save as parquet_to_jsonl.py
import argparse, json, os
from datasets import load_dataset, DatasetDict

def write_split(ds, out_jsonl, text_col, keep_newlines=False):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in ds:
            txt = ex[text_col]
            if txt is None:
                continue
            # if not keep_newlines:
            #     # optional: normalize hard line breaks if your data is very ragged
            #     txt = str(txt).replace("\r\n", "\n")
            # f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"messages": txt}, ensure_ascii=False) + "\n")
            # f.write(txt + "\n")
    print(f"Wrote {out_jsonl}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True,
                    help="One or more parquet file globs or URLs (HF data_files). "
                         "Use split tags like 'train:gs://.../*.parquet' 'test:...'. "
                         "If you pass bare paths, they are treated as train.")
    ap.add_argument("--text-column", default="text",
                    help="Column in Parquet that contains raw text")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to place JSONL files")
    ap.add_argument("--keep-newlines", action="store_true")
    args = ap.parse_args()

    # Build a data_files dict understood by datasets.load_dataset
    data_files = {}
    default_to_train = []
    for spec in args.parquet:
        splits = spec.split(os.pathsep)
        if len(splits) == 2:
            split, pattern = splits
            data_files.setdefault(split, [])
            data_files[split].append(pattern)
        else:
            default_to_train.append(spec)
    if default_to_train:
        data_files.setdefault("train", [])
        data_files["train"].extend(default_to_train)

    # Load as 'parquet' builder
    print("data_files:", data_files)
    ds: DatasetDict = load_dataset("parquet", data_files=data_files)

    for split in ds:
        out_jsonl = os.path.join(args.out_dir, f"{split}.jsonl")
        write_split(ds[split].shuffle(seed=1), out_jsonl, args.text_column, args.keep_newlines)

if __name__ == "__main__":
    main()
