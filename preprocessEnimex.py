import os, sys, re
sys.path.append("../")
from modules.utils import remove_hex, remove_multiple_whitespace


def remove_space_between_quantity(text):
    text = re.sub(r"([1-9][0-9]*)(in|inch|INCH|Inch|In)( |$)", r'\1inch ', text)
    text = re.sub(r"([1-9][0-9]*)(m|meter|M|METER|Meter)( |$)", r'\1m ', text)
    text = re.sub(r"([1-9][0-9]*)(mm|milimeter|MM|MILIMETER|Mm)( |$)", r'\1mm ', text)
    text = re.sub(r"([1-9][0-9]*)(cm|centimeter|CENTIMETER|CM|Cm)( |$)", r'\1ccm ', text)
    text = re.sub(r"([1-9][0-9]*)(pc|pcs|potong|pasang|Pasang|PCS|PC|Pc|Pcs)( |$)", r'\1pcs ', text)
    text = re.sub(r"([1-9][0-9]*)(y|year|thn|tahun|Year|Tahun)( |$)", r'\1tahun ', text)
    text = re.sub(r"([1-9][0-9]*)(k|kilo|Kilo|kg|kilogram|KG|Kg|Kilogram)( |$)", r'\1kg ', text)
    text = re.sub(r"([1-9][0-9]*)(g|gr|gram|G|Gr|GR|GRAM|Gram)( |$)", r'\1gr ', text)
    text = re.sub(r"([1-9][0-9]*)(l|liter|L|Liter|LITER)( |$)", r'\1l ', text)
    text = re.sub(r"([1-9][0-9]*)(ml|mililiter|ML|mL|Ml)( |$)", r'\1ml ', text)
    text = re.sub(r"([1-9][0-9]*) (in|inch|INCH|Inch|In)( |$)", r'\1inch ', text)
    text = re.sub(r"([1-9][0-9]*) (m|meter|M|METER|Meter)( |$)", r'\1m ', text)
    text = re.sub(r"([1-9][0-9]*) (mm|milimeter|MM|MILIMETER|Mm)( |$)", r'\1mm ', text)
    text = re.sub(r"([1-9][0-9]*) (cm|centimeter|CENTIMETER|CM|Cm)( |$)", r'\1ccm ', text)
    text = re.sub(r"([1-9][0-9]*) (pc|pcs|potong|pasang|Pasang|PCS|PC|Pc|Pcs)( |$)", r'\1pcs ', text)
    text = re.sub(r"([1-9][0-9]*) (y|year|thn|tahun|Year|Tahun)( |$)", r'\1tahun ', text)
    text = re.sub(r"([1-9][0-9]*) (k|kilo|Kilo|kg|kilogram|KG|Kg|Kilogram)( |$)", r'\1kg ', text)
    text = re.sub(r"([1-9][0-9]*) (g|gr|gram|G|Gr|GR|GRAM|Gram)( |$)", r'\1gr ', text)
    text = re.sub(r"([1-9][0-9]*) (l|liter|L|Liter|LITER)( |$)", r'\1l ', text)
    text = re.sub(r"([1-9][0-9]*) (ml|mililiter|ML|mL|Ml)( |$)", r'\1ml ', text)
    
    text = re.sub(
        r"([1-9][0-9]*) (yard|set|lembar|tablet|kaplet|buah|box|sachet|pasang|gb|watt)( |$)", r'\1\2 ', text)
    
    text = re.sub(r"([1-9][0-9]*) (x) ([1-9][0-9]*)", r'\1x\3', text)
    text = re.sub(r"([1-9][0-9]*) (in) ([1-9][0-9]*)", r'\1in\3', text)
    return text

def preprocessEnimex(text) :
    return text.replace("<", " <").replace(">", "> ")

infile = "../input/id-product-extraction/annotated-dataset-ecommerce-indonesia.txt"
outfile = "../input/id-product-extraction/ecommerce-enimex.txt"
with open(infile, 'r', encoding="ascii", errors='ignore') as f, open(outfile, 'w') as out:
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
        line = preprocessEnimex(line)
        line = line.replace("\n","")
        line = line.replace("\t","")
        line = remove_multiple_whitespace(line)
        line = remove_hex(line)
        line = remove_space_between_quantity(line)
        line = "\"" + line + "\""
        line += '\n'
        out.write(line)


