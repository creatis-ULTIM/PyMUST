# To export the ipynb to text files (for submission)
import json, os
#Copied from https://stackoverflow.com/questions/37797709/convert-json-ipython-notebook-ipynb-to-py-file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-input", "-i", type = str, help="Input file/folder", default = '.')

parser.add_argument("--noCode", help="Do not write the code",
                    action="store_true")
parser.add_argument("-output", "-o", type = str, help="Output", default = 'ExportedNB')

args = parser.parse_args()

writeCode = not args.noCode
writeMarkdown = True
writeAllMarkdown = True

files = []
if not os.path.exists(args.input):
    print("Input file/folder does not exist")
    exit()
elif os.path.isfile(args.input):
    files.append(args.input)
else:
    for file in os.listdir(args.input):
        if not file.endswith('.ipynb'):
            continue
        files.append(os.path.join(args.input, file))
if not os.path.exists(args.output):
    os.makedirs(args.output)

for file in files:
    code = json.load(open(file))
    py_file = open(f"{args.output}/{file.replace('ipynb', 'txt')}", "w+")

    for cell in code['cells']:
        if cell['cell_type'] == 'code' and writeCode:
            for line in cell['source']:
                py_file.write(line)
            py_file.write("\n")
        elif cell['cell_type'] == 'markdown' and writeMarkdown:
            py_file.write("\n")
            for line in cell['source']:
                if line and line[0] == "#" or writeAllMarkdown:
                    py_file.write(line)
            py_file.write("\n")

    py_file.close()