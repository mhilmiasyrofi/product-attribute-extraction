import os, sys, re

# from helper import remove_hex, remove_multiple_whitespace
import helper


def convert_raw_into_enimex(input_file:str, output_file:str):

    with open(input_file, 'r', encoding="ascii", errors='ignore') as f, open(output_file, 'w') as out:
        # for line in f.readlines():
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i][:-1]
            while i < len(lines) and line[-1] != "\"":
                i += 1
                if i < len(lines):
                    line += lines[i][:-1]
            i += 1  # for the next while
            # print(i)
            line = line[1:-1]
            line = line.replace("<", " <").replace(">", "> ")
            line = line.replace("\n", "")
            line = line.replace("\t", "")
            line = helper.remove_multiple_whitespace(line)
            line = helper.remove_hex(line)
            line = helper.remove_space_between_quantity(line)
            line = "\"" + line + "\""
            line += '\n'
            out.write(line)


if __name__ == "__main__" :

    input_file = "data/annotated-dataset-ecommerce-indonesia.txt"
    output_file = "data/ecommerce-enimex.txt"

    convert_raw_into_enimex(input_file=input_file, output_file=output_file)





