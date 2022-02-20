import sys
sys.path.append("../")


from modules.utils import remove_punctuation, remove_multiple_whitespace

def process(s):
    res = []
    for token in s[:-1] : # unprocess last sentence
        word = token.split("\t")[0]
        tag = token.split("\t")[1]
        word = remove_punctuation(word)
        word = remove_multiple_whitespace(word)

        if word != "" :
            res.append(word + "\t" + tag)
    return "".join(res)


if __name__ == "__main__" :
    infile = "../input/id-product-extraction/ecommerce-BIO.txt"
    outfile = "../input/id-product-extraction/ecommerce.txt"
    NON_ENTITY_TYPE = 'O'

    # print(infile)
    with open(infile, 'r', encoding="ascii", errors='ignore') as f, open(outfile, 'w') as out:
        l = 0
        s = []
        for line in f.readlines():
            if line[:-1] == "" :
                if l > 3 :
                    s = process(s)
                    out.write(s)
                    out.write("\n")
                l = 0
                s = []
            else :
                l += 1
                s.append(line)
        
