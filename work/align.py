import sys
import collections
import argparse
import functools

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--force', action='store_true', help='Force alignment to source side')
ap.add_argument('sfilename')
ap.add_argument('tfilename')
ap.add_argument('sstart', type=float)
ap.add_argument('tstart', type=float)
ap.add_argument('send', type=float)
ap.add_argument('tend', type=float)
args = ap.parse_args()

# Read data files
def read_data(f):
    for line in f:
        start, end, line = line.rstrip("\r\n").split("\t")
        yield float(start), float(end), line.strip()
sdata = list(read_data(open(args.sfilename)))
tdata = list(read_data(open(args.tfilename)))

# Linear transformation from s timestamps to t timestamps
def f(t):
    time_m = (args.tend-args.tstart)/(args.send-args.sstart)
    time_b = args.tstart-time_m*args.sstart
    return time_m*t+time_b

# Join two subtitles into one
def join(x, y):
    if x.endswith('...'):
        return f"{x.removesuffix('...')} {y}"
    elif x[-1] in '.,?!．，？！':
        return x + y
    else:
        return f'{x} / {y}'

def align(sdata, tdata):
    while len(sdata) > 0 and len(tdata) > 0:
        if f(sdata[0][1]) < tdata[0][0]:
            print(f'deleting: {sdata[0][2]}\t{f(sdata[0][0])}\t{f(sdata[0][1])}', file=sys.stderr)
            if args.force:
                print(f'{sdata[0][0]}\t{sdata[0][1]}\t{sdata[0][2]}\t')
            sdata.pop(0)
            continue
        elif tdata[0][1] < f(sdata[0][0]):
            print(f'deleting: {tdata[0][2]}\t{tdata[0][0]}\t{tdata[0][1]}', file=sys.stderr)
            tdata.pop(0)
            continue
        else:
            overlaps = []
            for (sk, tk) in [(1,1), (1,2), (2,1), (1,3), (3,1)]:
                if len(sdata) < sk or len(tdata) < tk: continue
                if args.force and sk != 1: continue
                sstart, send = f(sdata[0][0]), f(sdata[sk-1][1])
                tstart, tend = tdata[0][0], tdata[tk-1][1]
                overlap = 2 * (min(send, tend) - max(sstart, tstart)) / (send - sstart + tend - tstart)
                overlaps.append((overlap, sk, tk))
            (overlap, sk, tk) = max(overlaps)
            sline = functools.reduce(join, [s for (_, _, s) in sdata[:sk]])
            tline = functools.reduce(join, [t for (_, _, t) in tdata[:tk]])
            # arbitrarily use source-side timestamps
            print(f'{sdata[0][0]}\t{sdata[sk-1][1]}\t{sline}\t{tline}')
            sdata[:sk] = []
            tdata[:tk] = []
    if args.force:
        while len(sdata) > 0:
            print(f'{sdata[0][0]}\t{sdata[0][1]}\t{sdata[0][2]}\t')
            sdata.pop(0)

align(sdata, tdata)        
