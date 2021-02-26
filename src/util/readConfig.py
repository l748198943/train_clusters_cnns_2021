# -*- coding: utf-8 -*-

# read config file and return a dict
def readConfig(path):
    args = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            arg = line.split('=')
            if len(arg) == 2:
                # print(arg)
                args[arg[0]] = arg[1]

    return args 

