from os import path
import numpy as np
import os
import re


class ParFile:
    def __init__(self, filename):
        self.types = ["label", "fileform", "character", "real", "complex", "integer", "table"]
        self.linbr = ".+\&"

        if not filename.endswith(".par"):
            fname = path.join(filename, "rhd.par")
        self.f = open(filename, 'r')
        self.read()
        self.f.close()

    def conv_type(self, etype):
        if etype in ["character", "label"]:
            return np.str_
        elif etype == "real":
            return np.float32
        elif etype == "complex":
            return np.complex32
        elif etype == "integer":
            return np.int32
        elif etype == "fileform":
            return etype
        elif "table" == etype:
            return list

    def split_params(self, param):
        param = re.findall("[\S]+='[^\']+'|[\S]+", param.strip())
        params = {}
        for p in param:
            p = p.strip().split('=')
            p[0] = p[0].strip()
            params[p[0]] = re.findall("[^-']+", p[1])
            if not params[p[0]]:
                try:
                    del params[p[0]]
                except RuntimeError:
                    pass
            else:
                params[p[0]] = params[p[0]][0].strip()
        if 'b' in params:
            params['b'] = int(params['b'])
        if 'p' in params:
            params['p'] = int(params['p'])
        if 'd' in params:
            fshape = []
            for x in re.findall("\d+:\d+", params['d']):
                x1, x2 = x.split(':')
                fshape.append(int(x2) - int(x1) + 1)
            params['d'] = tuple(reversed(fshape))
        return params

    def get_entry(self, entry):
        while True:
            if re.match(self.linbr, self.line) is not None:
                self.line = re.match(self.linbr, self.line).group()[:-1] + self.f.readline()
            if re.match(self.linbr, self.line) is None:
                break
        etype, name, param = re.findall("(\w+) (\w+) ?(.*)$", self.line)[0]
        entry[name] = {}
        entry[name]['type'] = self.conv_type(etype)
        entry[name]['params'] = self.split_params(param)
        if 'b' in entry[name]['params']:
            if 'd' in entry[name]['params']:
                self.line = self.f.readline()
                if len(entry[name]['params']['d']) == 1:
                    entry[name]['data'] = entry[name]['type'](self.line.strip())
                else:
                    val = []
                    while True:
                        for i in self.line.split():
                            val.append(i)
                        if any(self.line.startswith(ty) for ty in self.types):
                            break
                        self.line = self.f.readline()
                        entry[name]['data'] = np.array(val, dtype=entry[name]['type']).reshape(entry[name]['params']
                                                                                               ['d'])
            else:
                self.line = self.f.readline()
                entry[name]['data'] = entry[name]['type'](self.line.strip())
        self.line = self.f.readline()
        self.line = self.line.strip()
        return entry

    def read(self):
        count = 0
        self.struc = {}
        for self.line in self.f:
            self.line = self.line.strip()
            if "fileform" in self.line:
                self.struc['Header'] = {}
                while "label" not in self.line:
                    if any(self.line.startswith(ty) for ty in self.types):
                        self.struc['Header'] = self.get_entry(self.struc['Header'])
                    else:
                        self.line = self.f.readline()
                        self.line = self.line.strip()
            elif "label" in self.line:
                while True:
                    if re.match(self.linbr, self.line) is not None:
                        self.line = re.match(self.linbr, self.line).group()[:-1] + self.f.readline()
                    if re.match(self.linbr, self.line) is None:
                        break
                self.line = self.f.readline()
                self.line = self.line.strip()
            elif any(self.line.startswith(ty) for ty in self.types[2:]):
                while True:
                    if len(self.line) == 0:
                        count += 1
                    else:
                        count = 0
                    if any(self.line.startswith(ty) for ty in self.types):
                        self.struc = self.get_entry(self.struc)
                    else:
                        self.line = self.f.readline()
                        self.line = self.line.strip()
                    if "label" in self.line or count == 4:
                        break

if __name__ == "__main__":
    parname = r"N:\Python\Data\d3gt57g44v50rsn01_400x400x188.par"
    par = ParFile(parname)
