'''
Convert ENAMEX Named-Entity annotated file to Stanford NLP format (token-based)
@Author yohanes.gultom@gmail
ENAMEX example (2 sentences):
Sementara itu Pengamat Pasar Modal <ENAMEX TYPE="PERSON">Dandossi Matram</ENAMEX> mengatakan, sulit bagi sebuah <ENAMEX TYPE="ORGANIZATION">kantor akuntan publik</ENAMEX> (<ENAMEX TYPE="ORGANIZATION">KAP</ENAMEX>) untuk dapat menyelesaikan audit perusahaan sebesar <ENAMEX TYPE="ORGANIZATION">Telkom</ENAMEX> dalam waktu 3 bulan.	1
<ENAMEX TYPE="ORGANIZATION">Telkom</ENAMEX> akan melakukan RUPS pada 30 Juli 2004 yang selain melaporkan kinerja 2003 juga akan meminta persetujuan untuk pemecahan nilai nominal saham atau stock split 1:2.	2
'''

import re
import sys
sys.path.append("../")

from modules.utils import remove_multiple_whitespace

NON_ENTITY_TYPE = 'O'

if __name__ == "__main__" : 
    infile = "../input/id-product-extraction/ecommerce-stanford.txt"
    outfile = "../input/id-product-extraction/ecommerce-BIO.txt"
    cur_type = NON_ENTITY_TYPE
    # print(infile)
    with open(infile, 'r', encoding="ascii", errors='ignore') as f, open(outfile, 'w') as out:
        prev = None
        prev_dot = False ## avoid printing double dot
        is_last = False
        for line in f.readlines():
            # line = remove_multiple_whitespace(line)
            tokens = line.split('\t')
            token, cur_type = tokens[0], tokens[1][:-1]
            if not token or token == "" :
                continue

            if len(token) > 2 and token[0] == "\"" and token[-1] == "\"":
                token = token[1:-1]
            elif len(token) > 1 and token[0] == "\"":
                token = token[1:]
            elif len(token) > 1 and token[-1] == "\"":
                token = token[:-1]

            if token == "\"" :
                if not prev_dot :
                    out.write("." + '\t' + NON_ENTITY_TYPE + '\n')
                    prev_dot = True
                    out.write('\n')
                prev = None
            else :
                token = token.lower()
                if token[-1] == "\"":
                    token = token[:-1]
                    is_last = True

                if cur_type == NON_ENTITY_TYPE:
                    out.write(token + '\t' + cur_type + '\n')
                else :
                    if not prev :
                        out.write(token + '\tB-' + cur_type + '\n')
                    else :
                        if prev == cur_type :
                            out.write(token + '\tI-' + cur_type + '\n')
                        else :
                            out.write(token + '\tB-' + cur_type + '\n')
                prev = cur_type
                prev_dot = False

                if is_last :
                    prev = None
                    if not prev_dot :
                        out.write("." + '\t' + NON_ENTITY_TYPE + '\n')
                        prev_dot = True
                        out.write('\n')
                    is_last = False


