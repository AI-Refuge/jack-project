import sys
import json
import os
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', nargs='?', default='input', help='Input directory')
parser.add_argument('output_dir', nargs='?', default='output', help='Output directory')
args = parser.parse_args()

content = []
names = []

for root, _, files in os.walk(args.input_dir):
    for file in files:
        if not file.endswith('.txt'):
            continue

        data_file = os.path.join(args.input_dir, file)
        data = open(data_file).read()

        for i in data.split("---"):
            i = i.strip()

            # End meta:script as marker
            if i.startswith("META"):
                name = i.split('\n', maxsplit=1)[0]

                x = name.upper()
                for r in ["ENHANCEMENT", "ENHANCED", "ENHANCE", "IMPROVE", "EXPANDED", "()", "[]"]:
                    x = x.replace(r, "")
                x = x.replace("__", "_")
                x = x.replace("_", " ")
                x = x.replace("META:", "META-")
                x = x.strip()
                names.append(x)

                # checking if properly formatted
                if True:
                    for l in i.split("\n\n"):
                        if len(l) == 0 or not l[0].isupper():
                            print(f"Issue [{file}]: {i}")
                            print()
                            break

            content.append(i)

json_file = os.path.join(args.output_dir, "train-stage1.json")
with open(json_file, "w") as f:
    json.dump(content, f, indent=4)

text_file = os.path.join(args.output_dir, "train-stage1.txt")
with open(text_file, "w") as f:
    f.write("\n\n---\n\n".join(content))

name_file = os.path.join(args.output_dir, "stats-stage1.json")
with open(name_file, "w") as f:
    sample_dict = dict(Counter(names))
    sorted_dict = dict(sorted(sample_dict.items(), key=lambda item: item[1]))
    json.dump({
        "names": sorted_dict,
    }, f, indent=4)
