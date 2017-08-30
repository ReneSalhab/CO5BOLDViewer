from collections import Mapping
from struct import Struct
from os import path
import numpy as np
import os
import re

_uint_from_bytes = Struct('>I')


class _EntryMapping(Mapping):
    def __init__(self, entries):
        self._entries = entries

    def __getitem__(self, key):
        for e in self._entries:
            if e.name == key:
                return e
        raise KeyError('%s' % key)

    def __iter__(self):
        for e in self._entries:
            yield e.name

    def __len__(self):
        return len(self._entries)

    @property
    def values(self):
        return self._entries

    @property
    def items(self):
        return [(e.name, e) for e in self._entries]

    def iteritems(self):
        for e in self._entries:
            yield (e.name, e)

    def itervalues(self):
        for e in self._entries:
            yield e


class Entry(object):
    def __init__(self, type, name, params, dtype, shape, data):
        self.type = type
        self.name = name
        self.params = params
        self.dtype = dtype
        self.shape = shape
        self.data = data
        if 'u' in params:
            self.unit = params['u']
        else:
            self.unit = ''

    # @property
    # def params(self):
    #     slist = [self.type, self.name]
    #     for k in ['b', 'd', 'u']:
    #         if k in self.params:
    #             v = self.params[k]
    #             if ' ' in v:
    #                 v = "'%s'" % v
    #             slist.append('%s=%s' % (k, v))
    #     return '<%s>' % (' '.join(slist))


class ParFile(_EntryMapping):
    def __init__(self, filename):
        self.types = ["label", "fileform", "character", "real", "complex", "integer", "table"]
        self.linbr = ".+\&"

        with open(filename, 'r') as self.f:
            self._read()

    def __repr__(self):
        slist = ["ParFile"]
        slist.append("entries={0}".format(len(self._entries)))
        return "<{0}>".format(' '.join(slist))

    def _conv_type(self, etype):
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
        else:
            raise TypeError("Unknown type found!")

    def _split_params(self, param):
        param = re.findall("[\S]+='[^\']+'|[\S]+", param.strip())
        params = {}
        for p in param:
            p = p.strip().split('=')
            p[0] = p[0].strip()
            if len(p) == 2:
                params[p[0]] = re.findall("[^-']+", p[1])
            elif len(p) > 2:
                params[p[0]] = re.findall("[^-']+", " ".join(p[1:]))
            else:
                print("Found invalid entry in line {0}. Maybe single quotation mark is missing.".format(self.lineno))
                continue
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

    def _get_entry(self):
        while re.match(self.linbr, self.line) is not None:
            # better method for combining description lines. Errors like several "&" are considered
            self.line = "".join(re.match(self.linbr, self.line).group().split("&")) + self._readline()

        etype, name, param = re.findall("(\w+) (\w+) ?(.*)$", self.line)[0]
        dtype = self._conv_type(etype)
        params = self._split_params(param)

        if 'b' in params:
            if 'd' in params:
                shape = params['d']
                self.line = self._readline()
                if len(shape) == 1:
                    data = dtype(self.line.strip())
                else:
                    val = []
                    while True:
                        for i in self.line.split():
                            val.append(i)
                        self.line = self._readline()
                        if any(self.line.startswith(ty) for ty in self.types):
                            break

                    data = np.array(val, dtype=dtype).reshape(shape)
            else:
                shape = None
                self.line = self._readline()
                data = dtype(self.line.strip())
            self.line = self._readline()
            self.line = self.line.strip()
        else:
            self.line = self._readline()
            self.line = self.line.strip()
            return

        return Entry(type=etype, name=name, params=params, dtype=dtype, shape=shape, data=data)

    def _read(self):
        count = 0
        entries = []
        self.lineno = 0
        for self.line in self.f:
            self.lineno += 1
            self.line = self.line.strip()
            if "fileform" in self.line:
                # self.header = {}
                while "label" not in self.line:
                    self.header = []
                    if any(self.line.startswith(ty) for ty in self.types):
                        self.header.append(self._get_entry())
                    else:
                        self.line = self._readline()
                        self.line = self.line.strip()
            elif "label" in self.line:
                while True:
                    if re.match(self.linbr, self.line) is not None:
                        self.line = re.match(self.linbr, self.line).group()[:-1] + self._readline()
                    if re.match(self.linbr, self.line) is None:
                        break
                self.line = self._readline()
                self.line = self.line.strip()
            elif any(self.line.startswith(ty) for ty in self.types[2:]):
                while True:
                    if len(self.line) == 0:
                        count += 1
                    else:
                        count = 0
                    if any(self.line.startswith(ty) for ty in self.types):
                        entries.append(self._get_entry())
                    else:
                        self.line = self._readline()
                        self.line = self.line.strip()
                    if "label" in self.line or count == 4:
                        break
        super(ParFile, self).__init__(entries)

    def _readline(self):
        self.lineno += 1
        return self.f.readline()

if __name__ == "__main__":
    parname = r"N:\Python\Data\d3gt57g44v50rsn01_400x400x188.par"
    par = ParFile(parname)
