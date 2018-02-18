#!/usr/bin/python
# -*- coding: utf-8 -*-

class Subjective:

    def __init__(self, internal_id, begin_posi, end_posi, token, subj_id, polar, related):
        self.internal_id = internal_id
        self.begin_posi = begin_posi
        self.end_posi = end_posi
        self.token = token
        self.subj_id = subj_id
        self.polar = polar
        self.related = related

def getInfosFromLine(line):
    eles = line.split('\t')

    s = Subjective(eles[1],eles[2],eles[3],eles[4],eles[5],eles[6],eles[7])

    return s

def getAllSubjectivesFromFile(csvFile):

    subjs = []

    f = open(csvFile)
    for line in f.readlines():
        line = line.strip()
        if line.startswith('subjective'):
            s = getInfosFromLine(line)
            subjs.append(s)

    return subjs

def writeAllSubjectives():

    products = ['coffeemachine','cutlery','microwave','toaster','trashcan','vacuum','washer']

    anno = ['a1','a2']

    for an in anno:

        fw1 = open('../paraphrases/subjective-pos-'+an+'.txt','w')
        fw2 = open('../paraphrases/subjective-neg-'+an+'.txt','w')

        for pro in products:
            subjs = getAllSubjectivesFromFile('../files/en-'+pro+'-'+an+'.csv')

            for s in subjs:
                if s.polar == 'positive':
                    fw1.write(s.token+'\n')
                elif s.polar == 'negative':
                    fw2.write(s.token+'\n')


if __name__ == '__main__':

    writeAllSubjectives()

    """
    subjs = getAllSubjectivesFromFile('../files/en-coffeemachine-a1.csv')

    for s in subjs:
        print s.token
        print s.polar
        print s.begin_posi
    """