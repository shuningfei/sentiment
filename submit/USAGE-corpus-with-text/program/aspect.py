#!/usr/bin/python
# -*- coding: utf-8 -*-

class Aspect:


    def __init__(self, internal_id, begin_posi, end_posi, token, aspect_id, related):
        self.internal_id = internal_id
        self.begin_posi = begin_posi
        self.end_posi = end_posi
        self.token = token
        self.aspect_id = aspect_id
        self.related = related


def getInfosFromLine(line):
    eles = line.split('\t')

    a = Aspect(eles[1],eles[2],eles[3],eles[4],eles[5],eles[7])

    return a

def getAllAspectsFromFile(csvFile):

    aspects = []

    f = open(csvFile)
    for line in f.readlines():
        line = line.strip()
        if line.startswith('aspect'):
            a = getInfosFromLine(line)
            aspects.append(a)

    return aspects

def buildCorefAspectDict(relFile,csvFile):

    aspects = getAllAspectsFromFile(csvFile)
    d = {}

    f = open(relFile)
    for line in f.readlines():
        line = line.strip()
        if line.startswith('COREF'):
            eles = line.split('\t')
            orignial_aspe_id = eles[3] # coffee maker
            target_aspe_id = eles[2] # this

            for a1 in aspects:
                for a2 in aspects:
                    if a1.aspect_id == orignial_aspe_id and a2.aspect_id == target_aspe_id:
                        d[a1] = a2

    return d

if __name__ == '__main__':

    """

    aspects = getAllAspectsFromFile('../files/en-coffeemachine-a1.csv')

    for a in aspects:
        print a.token
    """

    d = buildCorefAspectDict('../files/en-coffeemachine-a1.rel','../files/en-coffeemachine-a1.csv')

    for ele in d:
        print ele.internal_id
        print ele.token
        print (ele.begin_posi,ele.end_posi)
        print