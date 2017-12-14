# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2013

@author: Kolja Glogowski

Modified: René Salhab (10.03.2017)
    - Put functions into corresponding classes
    - Introduced compatibility with .eos files
"""

import os
import re
import collections
import numpy as np
from struct import Struct

_uint_from_bytes = Struct('>I')

class _EntryMapping(collections.Mapping):
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

    def values(self):
        return self._entries

    def items(self):
        return [(e.name, e) for e in self._entries]

    def iteritems(self):
        for e in self._entries:
            yield (e.name, e)

    def itervalues(self):
        for e in self._entries:
            yield e


class Entry(object):
    def __init__(self, fd, pos, type, name, params, dtype, shape):
        self._fd = fd
        self.pos = pos
        self.type = type
        self.name = name
        self.params = params
        self.dtype = dtype
        self.shape = shape

    def __repr__(self):
        slist = [self.type, self.name]
        for k in ['b', 'd', 'u']:
            if k in self.params:
                v = self.params[k]
                if ' ' in v:
                    v = "'%s'" % v
                slist.append('%s=%s' % (k, v))
        return '<%s>' % (' '.join(slist))

    @property
    def data(self):
        self._fd.seek(self.pos)
        a = self._read_array()
        if self.type != 'character':
            return a.reshape(self.shape) if self.shape != None else a[0]
        else:
            if self.shape == None:
                return a[0].rstrip()
            elif len(self.shape) == 1:
                return [s.rstrip() for s in a]
            else:
                return a.reshape(self.shape)

    def _read_array(self):
        dt = np.dtype(self.dtype).newbyteorder('>')
        nbytes = self._read_block_size()
        data = np.fromfile(self._fd, dtype=dt, count=nbytes // dt.itemsize)
        if nbytes != self._read_block_size():
            raise IOError('error reading array')
        return data.astype(data.dtype.newbyteorder('='))

    def _read_block_size(self):
        data = self._fd.read(4)
        if len(data) != 4:
            raise EOFError()
        return _uint_from_bytes.unpack(data)[0]


class Block(_EntryMapping):
    def __init__(self, pos, params, entries):
        self.pos = pos
        self.params = params
        super(Block, self).__init__(entries)

    def __repr__(self):
        slist = [ 'block' ]
        slist.append('entries=%d' % len(self._entries))
        return '<%s>' % (' '.join(slist))


class Box(_EntryMapping):
    def __init__(self, pos, params, entries):
        self.pos = pos
        self.params = params
        super(Box, self).__init__(entries)

    def __repr__(self):
        slist = [ 'box' ]
        slist.append('entries=%d' % len(self._entries))
        return '<%s>' % (' '.join(slist))


class DataSet(_EntryMapping):
    def __init__(self, pos, params, entries, box):
        self.pos = pos
        self.params = params
        self.box = box
        super(DataSet, self).__init__(entries)

    def __repr__(self):
        slist = [ 'dataset' ]
        slist.append('entries=%d' % len(self._entries))
        slist.append('boxes=%d' % len(self.box))
        return '<%s>' % (' '.join(slist))

class File(_EntryMapping):
    def __init__(self, filename):
        _, fend = os.path.splitext(filename)
        self._fd = open(os.path.abspath(filename), 'rb')

        self._re_params = re.compile(r"\w+=(?:(?:'[^']*')|(?:\S+))")
        self._re_desc = re.compile(r"^(\w+) (\w+) ?(.*)$")
        self._re_dims = re.compile('\d+:\d+')

        self._fd.seek(0)
        ff, uio, self.header = self._parse_descriptor()
        if ff != 'fileform' or uio != 'uio':
            raise IOError('unknown file format')

        entries = []
        if fend == ".eos":
            self.block = []
            self.dataset = None
        else:
            self.block = None
            self.dataset = []

        for x in self._read_file_entries():
            if isinstance(x, DataSet):
                self.dataset.append(x)
            elif isinstance(x, Entry):
                entries.append(x)
            elif isinstance(x, Block):
                self.block.append(x)

        super(File, self).__init__(entries)

    def __repr__(self):
        slist = [ 'uio' ]
        slist.append('entries=%d' % len(self._entries))
        if isinstance(self.block, list):
            slist.append('blocks=%d' % len(self.block))
        elif isinstance(self.dataset, list):
            slist.append('datasets=%d' % len(self.dataset))
        return '<%s>' % (' '.join(slist))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fd.close()

    def close(self):
        self._fd.close()

    @property
    def closed(self):
        return self._fd.closed

    @property
    def name(self):
        return self._fd.name

    def _read_file_entries(self):
        entries = []
        while True:
            try:
                entry = self._read_entry()
            except EOFError:
                break
            if entry.type == 'label' and entry.name == 'dataset':
                ds_box = []
                ds_entries = []
                for ei in self._read_dataset_entries():
                    if isinstance(ei, Box):
                        ds_box.append(ei)
                    elif isinstance(ei, Entry):
                        ds_entries.append(ei)
                ds = DataSet(pos=self._fd.tell(), params=entry.params, entries=ds_entries, box=ds_box)
                entries.append(ds)
            elif entry.type == 'label' and entry.name == 'block':
                ds_entries = []
                for ei in self._read_box_entries():
                    if isinstance(ei, Entry):
                        ds_entries.append(ei)
                ds = Block(pos=self._fd.tell(), params=entry.params, entries=ds_entries)
                entries.append(ds)
            else:
                entries.append(entry)
                self._skip_block()
        return entries

    def _read_dataset_entries(self):
        while True:
            entry = self._read_entry()
            if entry.type == 'label':
                if entry.name == 'enddataset':
                    break
                elif entry.name == 'box':
                    yield Box(pos=self._fd.tell(), params=entry.params, entries=self._read_box_entries())
                elif entry.name == 'block':
                    yield Block(pos=self._fd.tell(), params=entry.params, entries=self._read_box_entries())
            else:
                yield entry
                self._skip_block()

    def _read_box_entries(self):
        box = []
        while True:
            entry = self._read_entry()
            if entry.type == 'label' and (entry.name == 'endbox' or entry.name == 'endblock'):
                break
            box.append(entry)
            self._skip_block()
        return box

    def _read_entry(self):
        etype, name, params = self._parse_descriptor()
        if etype == 'table':
            raise IOError('uio files containing tables are not supported')
        pos = self._fd.tell()
        dtype = None
        if 'b' in params:
            nbytes = int(params['b'])
            if etype == 'real':
                dtype = 'f%d' % nbytes
            elif etype == 'integer':
                dtype = 'i%d' % nbytes
            elif etype == 'character':
                dtype = 'S%d' % nbytes
            elif etype == 'complex':
                dtype = 'c%d' % nbytes
        shape = None
        if 'd' in params:
            fshape = []
            for x in self._re_dims.findall(params['d']):
                x1, x2 = x.split(':')
                fshape.append(int(x2) - int(x1) + 1)
            shape = tuple(reversed(fshape))
        return Entry(fd=self._fd, pos=pos, type=etype, name=name, params=params, dtype=dtype, shape=shape)

    def _read_string(self):
        s = self._read_block()
        while s.rstrip().endswith(b'&'):
            s = s.rstrip()[:-1] + self._read_block()
        return s.rstrip()

    def _read_block(self):
        nbytes = self._read_block_size()
        data = self._fd.read(nbytes)
        if nbytes != self._read_block_size():
            raise IOError('error reading block')
        return data

    def _skip_block(self):
        nbytes = self._read_block_size()
        self._fd.seek(nbytes, 1)
        if nbytes != self._read_block_size():
            raise IOError('error skipping block')

    def _read_block_size(self):
        data = self._fd.read(4)
        if len(data) != 4:
            raise EOFError()
        return _uint_from_bytes.unpack(data)[0]

    def _parse_descriptor(self):
        m = self._re_desc.match(self._read_string().decode())
        if not m:
            return None
        etype, name, pstr = m.groups()
        params = {}
        for it in self._re_params.findall(pstr):
            key, value = it.split('=', 1)
            params[key] = value.strip("'")
        return etype, name, params

if __name__ == '__main__':
    from numpy import *

    fname = os.path.join("Z:\cobold\scratchy\job_d3gt57g44v100rsn01_400x400x188", "rhd010.mean")

    f = File(fname)
