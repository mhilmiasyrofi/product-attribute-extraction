import os, sys, re

# from helper import remove_hex, remove_multiple_whitespace
import helper


def convert_raw_to_enimex(input_file:str, output_file:str):

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


def convert_enimex_to_stanford(input_file:str, output_file:str):
    '''
    Convert ENAMEX Named-Entity annotated file to Stanford NLP format (token-based)
    @Author yohanes.gultom@gmail
    ENAMEX example (2 sentences):
    Sementara itu Pengamat Pasar Modal <ENAMEX TYPE="PERSON">Dandossi Matram</ENAMEX> mengatakan, sulit bagi sebuah <ENAMEX TYPE="ORGANIZATION">kantor akuntan publik</ENAMEX> (<ENAMEX TYPE="ORGANIZATION">KAP</ENAMEX>) untuk dapat menyelesaikan audit perusahaan sebesar <ENAMEX TYPE="ORGANIZATION">Telkom</ENAMEX> dalam waktu 3 bulan.	1
    <ENAMEX TYPE="ORGANIZATION">Telkom</ENAMEX> akan melakukan RUPS pada 30 Juli 2004 yang selain melaporkan kinerja 2003 juga akan meminta persetujuan untuk pemecahan nilai nominal saham atau stock split 1:2.	2
    '''

    START_PATTERN = re.compile(r'^(.*?)<ENAMEX$', re.I)
    END_SINGLE_PATTERN = re.compile(r'^TYPE="(.*?)">(.*?)</ENAMEX>(.*?)$', re.I)
    TYPE_PATTERN = re.compile(r'^TYPE="(.*?)">(.*?)$', re.I)
    END_MULTI_PATTERN = re.compile(r'^(.*?)</ENAMEX>(.*?)$', re.I)
    EOS_PATTERN = re.compile(r'^([^<>]*)\.?\t(\d+)$', re.I)
    NON_ENTITY_TYPE = 'O'


    def check_and_process_eos(token):
        match = re.match(EOS_PATTERN, token)
        if match:
            out.write(match.group(1) + '\t' + cur_type + '\n')
            out.write('.' + '\t' + cur_type + '\n')
            out.write('\n')
            return True
        return False


    cur_type = NON_ENTITY_TYPE
    # print(infile)
    with open(input_file, 'r', encoding="ascii", errors='ignore') as f, open(output_file, 'w') as out:
        # for line in f.readlines():
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i][:-1]
            i += 1  # for the next while
            line = helper.remove_multiple_whitespace(line)
            for token in line.strip().split(' '):
                token = token.strip()
                if not token:
                    continue

                match = re.match(START_PATTERN, token)
                if match:
                    if match.group(1):
                        out.write(match.group(1) + '\t' +
                                  NON_ENTITY_TYPE + '\n')
                    continue

                match = re.match(END_SINGLE_PATTERN, token)
                if match:
                    out.write(match.group(2) + '\t' + match.group(1) + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(match.group(3)):
                        out.write(match.group(3) + '\t' + cur_type + '\n')
                    continue

                match = re.match(TYPE_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    out.write(match.group(2) + '\t' + cur_type + '\n')
                    continue

                match = re.match(END_MULTI_PATTERN, token)
                if match:
                    out.write(match.group(1) + '\t' + cur_type + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(match.group(2)):
                        out.write(match.group(2) + '\t' + cur_type + '\n')
                    continue

                if check_and_process_eos(token):
                    continue

                out.write(token + '\t' + cur_type + '\n')


if __name__ == "__main__" :

    input_file = "data/annotated-dataset-ecommerce-indonesia.txt"
    output_file = "data/ecommerce-enimex.txt"

    convert_raw_to_enimex(input_file=input_file, output_file=output_file)

    input_file = output_file
    output_file = "data/ecommerce-stanford.txt"

    convert_enimex_to_stanford(input_file=input_file, output_file=output_file)






