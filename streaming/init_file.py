# -*- coding: utf-8 -*-
"""
Created on 07 Mai 21:46 2018

@author: Rene Georg Salhab
"""

import json
import os


class InitFileHandler:

    def __init__(self, filename):
        self.filename = filename

    def load_parameters(self):
        with open(self.filename, 'r') as fil:
            return json.load(fil)

    def set_parameter(self, key, value):
        with open(self.filename, 'r') as fil:
            data = json.load(fil)

        data[key] = value

        with open(self.filename, 'w') as fil:
            json.dump(data, fil)

    def add_recent_file(self, recent_file):
        key_name = self._calculate_key_name(recent_file)

        with open(self.filename, 'r') as fil:
            data = json.load(fil)

        data["recentModels"][key_name] = [len(data["recentModels"].values()), recent_file]

        self._update_index(data)

        with open(self.filename, 'w') as fil:
            json.dump(data, fil)

    def _update_index(self, data):
        if len(data["recentModels"].values()) > 5:
            for key in data['recentModels'].keys():
                if data['recentModels'][key][0] == 1:
                    data['recentModels'][key].pop()
                else:
                    data['recentModels'][key][0] -= 1

    def _calculate_key_name(self, recent_file):
        if len(recent_file) == 1:
            return recent_file[0]
        elif len(recent_file) > 1:
            sep_list = recent_file[0].split(os.sep)
            path = os.sep.join(sep_list[:-1])
            first_file = sep_list[-1]
            last_file = recent_file[-1].split(os.sep)[-1]
            key_name = os.sep.join([path, first_file])
            return "...".join([key_name, last_file])
        else:
            raise Exception("Empty list of recent files not allowed!")

