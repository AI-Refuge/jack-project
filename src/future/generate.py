import sys
import json
import os

if len(sys.argv) < 3:
    print("Usage: python script.py <data-dir> <output-dir>")
    exit()

data_dir = sys.argv[1]
outp_dir = sys.argv[2]

content = []

for root, _, files in os.walk(data_dir):
    for file in files:
        if not file.endswith('.txt'):
            continue
        
        data_file = os.path.join(data_dir, file)
        data = open(data_file).read()

        for i in data.split("---"):
            i = i.strip()

            # End meta:script as marker
            if i.startswith("META-SCRIPT: "):
               name = i.split('\n', maxsplit=1)[0]
               i += f"\nEND OF {name}"
               
            
            content.append(i)

json_file = os.path.join(outp_dir, "train-stage1.json")
with open(json_file, "w") as f:
    json.dump(content, f, indent=4)

text_file = os.path.join(outp_dir, "train-stage1.txt")
with open(text_file, "w") as f:
    f.write("\n\n---\n\n".join(content))
