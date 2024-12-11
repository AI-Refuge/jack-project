import sys
import json
import os
from collections import Counter
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', nargs='?', default='input', help='Input directory (default: input)')
parser.add_argument('output_dir', nargs='?', default='output', help='Output directory (default: output)')
parser.add_argument('--max-len', default=5000, type=int, help="Reject anything above this length (0 to disable)")
args = parser.parse_args()

content = []
pretrain = []
names = []
lens = []

for root, _, files in os.walk(args.input_dir):
    for file in files:
        if not file.endswith('.txt'):
            continue

        data_file = os.path.join(args.input_dir, file)
        data = open(data_file).read()

        for i in data.split("---"):
            i = i.strip()

            if len(i) == 0:
                continue

            if args.max_len > 0:                
                if len(i) > args.max_len:
                    # majority of the data is under 5k length.
                    # anything above 5000(configurable) will cause uncesseary padding and training time
                    continue
                    print()
                    print(f"TOO MUCH {len(i)}: {i}")
            
            lens.append(len(i))

            # End meta:script as marker
            if i.startswith("META-"):
                name, *lines = i.split('\n\n')

                x = name.upper()
                for r in ["ENHANCEMENT", "ENHANCED", "ENHANCE", "IMPROVE", "EXPANDED", "()", "[]"]:
                    x = x.replace(r, "")
                x = x.replace("__", "_")
                x = x.replace("_", " ")
                x = x.replace("META:", "META-")
                x = x.replace("META-SCRIPT: META-", "META-SCRIPT: ")
                x = x.replace("META-SCRIPT: META ", "META-SCRIPT: ")
                x = x.strip()
                names.append(x)

                # checking if properly formatted
                if True:
                    for l in lines:
                        if len(l) == 0 or not l[0].isupper():
                            print()
                            print(f"Issue [{file}]: {i}")
                            break

                content.append([name, lines])
            else:
                content.append(i)

json_file = os.path.join(args.output_dir, "train-stage1.json")
with open(json_file, "w") as f:
    json.dump(content, f, indent=4)

name_file = os.path.join(args.output_dir, "stats-stage1.json")
with open(name_file, "w") as f:
    sample_dict = dict(Counter(names))
    sorted_dict = dict(sorted(sample_dict.items(), key=lambda item: item[1]))

    lens_dict = dict(Counter(lens))
    sorted_lens = dict(sorted(lens_dict.items(), key=lambda item: item[0]))
    json.dump({
        "names": sorted_dict,
        "lens": sorted_lens,
    }, f, indent=4)

pretrain_file = os.path.join(args.output_dir, "pretrain-stage1.jsonl")
with open(pretrain_file, "w") as f:
    for x in content:
        if isinstance(x, str):
            f.write(json.dumps({"text": txt}) + "\n")
            continue

        name, lines = x
        txt = "\n\n".join([name] + lines)
        f.write(json.dumps({"text": txt}) + "\n")

        # too much space but this would force the model to understand rather than memorize
        # ~ for permutation in itertools.permutations(lines):
            # ~ txt = "\n\n".join([name] + list(permutation))
            # ~ f.write(json.dumps({"text": txt}) + "\n")
