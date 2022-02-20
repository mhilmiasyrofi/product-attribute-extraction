'''
Convert ENAMEX Named-Entity annotated file to Stanford NLP format (token-based)
@Author yohanes.gultom@gmail
ENAMEX example (2 sentences):
Sementara itu Pengamat Pasar Modal <ENAMEX TYPE="PERSON">Dandossi Matram</ENAMEX> mengatakan, sulit bagi sebuah <ENAMEX TYPE="ORGANIZATION">kantor akuntan publik</ENAMEX> (<ENAMEX TYPE="ORGANIZATION">KAP</ENAMEX>) untuk dapat menyelesaikan audit perusahaan sebesar <ENAMEX TYPE="ORGANIZATION">Telkom</ENAMEX> dalam waktu 3 bulan.	1
<ENAMEX TYPE="ORGANIZATION">Telkom</ENAMEX> akan melakukan RUPS pada 30 Juli 2004 yang selain melaporkan kinerja 2003 juga akan meminta persetujuan untuk pemecahan nilai nominal saham atau stock split 1:2.	2
'''

import sys
import re

sys.path.append("../")

from modules.utils import remove_multiple_whitespace

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

if __name__ == "__main__" :
    infile = "../input/id-product-extraction/ecommerce-enimex.txt"
    outfile = "../input/id-product-extraction/ecommerce-stanford.txt"
    cur_type = NON_ENTITY_TYPE
    # print(infile)
    with open(infile, 'r', encoding="ascii", errors='ignore') as f, open(outfile, 'w') as out:
        # for line in f.readlines():
        lines = f.readlines()
        i = 0
        while i < len(lines) :
            line = lines[i][:-1]
            i += 1 # for the next while
            line = remove_multiple_whitespace(line)
            for token in line.strip().split(' '):
                token = token.strip()
                if not token:
                    continue

                match = re.match(START_PATTERN, token)
                if match:
                    if match.group(1):
                        out.write(match.group(1) + '\t' + NON_ENTITY_TYPE + '\n')
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
