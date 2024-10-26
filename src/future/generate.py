import sys
import json
import os
from collections import Counter

if len(sys.argv) < 3:
    print("Usage: python script.py <data-dir> <output-dir>")
    exit()

data_dir = sys.argv[1]
outp_dir = sys.argv[2]

content = []
names = []

for root, _, files in os.walk(data_dir):
    for file in files:
        if not file.endswith('.txt'):
            continue

        data_file = os.path.join(data_dir, file)
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

json_file = os.path.join(outp_dir, "train-stage1.json")
with open(json_file, "w") as f:
    json.dump(content, f, indent=4)

text_file = os.path.join(outp_dir, "train-stage1.txt")
with open(text_file, "w") as f:
    f.write("\n\n---\n\n".join(content))

name_file = os.path.join(outp_dir, "names-stage1.txt")
with open(name_file, "w") as f:
    sample_dict = dict(Counter(names))
    sorted_dict = dict(sorted(sample_dict.items(), key=lambda item: item[1]))
    json.dump(sorted_dict, f, indent=4)
