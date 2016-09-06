# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 12:09:39 2014

@author: René Georg Salhab
"""

import numpy as np
import re

def stopif(expr):
    if expr:
        raise StopIteration
    return True

def read_opta(filename):
    opatb = []
    with open(filename,"r") as f:
        for line in f:
            if "NT\n" in line:
                for line1 in f:
                    if "*" in line1: break
                    dimT = np.int16(np.char.strip(line1))
            if "NP\n" in line:
                for line1 in f:
                    if "*" in line1: break
                    dimP = np.int16(np.char.strip(line1))
            if " NBAND\n" in line:
                for line1 in f:
                    if "*" in line1: break
                    dimBAND = np.int16(np.char.strip(line1)) + 1
            if "TABT\n" in line:
                temptb = np.array(list(re.split("  | |\n",line1) for line1 in f if
                                       stopif("*" in line1)))
            if "TABP\n" in line:
                presstb = np.array(list(re.split("  | |\n",line1) for line1 in f if
                                       stopif("*" in line1)))
            if "log10 P" in line:
                next(f)
                for line1 in f:
                    if "log10 P" in line1:
                        next(f)
                        continue
                    elif '*' in line1:
                        break
                    opatb.append(re.split(" |\n",line1))
    temptb = np.array([item for sublist in temptb for item in sublist if item != '']).astype(float)
    presstb = np.array([item for sublist in presstb for item in sublist if item != '']).astype(float)
    opatb = np.array([a.strip() for sublist in opatb for a in sublist if a != '']).astype(float)
    opatb = opatb.reshape(dimP,dimBAND,dimT)
    return temptb, presstb, opatb

if __name__ == '__main__':
    pass
