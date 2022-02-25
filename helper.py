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
