import string
import re

def remove_hex(text):
    """
    Remove Hex
    Example: 
    "\xe3\x80\x90Ramadan\xe3\x80\x91Dompet wanita multi-fungsi gesper dompet multi-card"
    """
    res = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i+1 < len(text) and text[i+1] == "x":
            i += 3
            res.append(" ")
        else:
            res.append(text[i])
        i += 1
    # text = text.encode('utf-8')
    # text = text.encode('ascii', 'ignore')
    # text = text.encode('ascii', errors='ignore')
    # text = unicode(text)
    # text = re.sub(r'[^\x00-\x7f]', r'', text)
    # filter(lambda x: x in printable, text)
    return "".join(res)


def remove_multiple_whitespace(text):
    """
    remove multiple whitespace
    it covers tabs and newlines also
    """
    return re.sub(' +', ' ', text.replace('\n', ' ').replace('\t', ' ')).strip()


def remove_punctuation(text):
    """
    Removing punctuations
    """
    return re.sub(r'[^\w\s]', r' ', text)


def remove_space_between_quantity(text):
    """
    200 ml -> 200ml
    3 kg -> 3kg
    200 x 200 -> 200x200
    3 in 1 -> 3in1
    Example: "Double Tape DOUBLE FOAM TAPE 55 mm 45 m 45 makan   2000 x 2000 scs"
    """
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
    
    text = re.sub(r"([1-9][0-9]*) (yard|set|lembar|tablet|kaplet|buah|box|sachet|pasang|gb|watt)( |$)", r'\1\2 ', text)
    
    text = re.sub(r"([1-9][0-9]*) (x) ([1-9][0-9]*)", r'\1x\3', text)
    text = re.sub(r"([1-9][0-9]*) (in) ([1-9][0-9]*)", r'\1in\3', text)
    return text
