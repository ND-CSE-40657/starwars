import itertools
import fileinput
import re

def isplit(iterable, pred):
    buf = []
    for x in iterable:
        if pred(x):
            yield buf
            buf.clear()
        else:
            buf.append(x)
    yield buf

def convert_timestamp(s):
    m = re.match(r"(\d+):(\d+):(\d+),(\d+)", s)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float(m.group(4))/1000

def clean_line(s):
    s = re.sub(r"<.*?>", "", s)
    s = re.sub(r"^\s*-\s*", "", s)
    s = " ".join(s.split())
    return s

for title in isplit(fileinput.input(), lambda s: s.strip() == ""):
    if len(title) == 0: continue
    num = int(title[0])
    m = re.match("(\S+) --> (\S+)", title[1])
    start = convert_timestamp(m.group(1))
    end = convert_timestamp(m.group(2))
    lines = " ".join(clean_line(line) for line in title[2:])
    print("{}\t{}\t{}".format(start, end, lines))
        
