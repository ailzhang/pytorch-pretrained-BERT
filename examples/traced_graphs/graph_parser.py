import re

swag = 'swag_full'

aten_funcs = set()
prim_funcs = set()

def find_idx(prefix, suffix, line, unique_set):
    start_idx = line.find(prefix)
    if start_idx >= 0:
        end_idx = line[start_idx:].find(suffix)
        func_name = line[start_idx + 6 : start_idx + end_idx]
        unique_set.add(func_name)

def parse(filename):
    global aten_funcs, prim_funcs_
    with open(filename, 'r') as f:
        for line in f.readlines():
            find_idx('aten::', '\\n', line, aten_funcs)
            find_idx('prim::', '\\n', line, prim_funcs)

if __name__ == '__main__':
    parse(swag)
    print('Torch\n\n', '\n'.join(aten_funcs))
    print('Prim\n\n', '\n'.join(prim_funcs))
