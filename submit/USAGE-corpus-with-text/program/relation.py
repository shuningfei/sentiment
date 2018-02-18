#!/usr/bin/python
# -*- coding: utf-8 -*-

from aspect import *
from subjective import *

class Relation:

    """
    if relation_type is TARG-SUBJ: ele1 is aspect, ele2 is subjective
    if relation_type is COREF: ele1 is aspect, ele2 is also aspect

    """

    def __init__(self, internal_id, relation_type, ele1, ele2):

        self.internal_id = internal_id
        self.relation_type = relation_type
        self.ele1 = ele1
        self.ele2 = ele2

def getInfosFromLine(line, aspects_list, subjs_list):
    eles = line.split('\t')
    relation_type = eles[0]
    internal_id = eles[1]
    ele1_id = eles[2]
    ele2_id = eles[3]

    ele1 = None
    ele2 = None
    r = None

    if relation_type == "TARG-SUBJ":
        for aspect in aspects_list:
            if aspect.aspect_id == ele1_id:
                ele1 = aspect

        for subj in subjs_list:
            if subj.subj_id == ele2_id:
                ele2 = subj

        if ele1!=None and ele2!=None:
            r = Relation(internal_id,relation_type,ele1,ele2)

    elif relation_type == "COREF":
        for aspect in aspects_list:
            if aspect.aspect_id == ele1_id:
                ele1 = aspect
            elif aspect.aspect_id == ele2_id:
                ele2 = aspect

        if ele1!=None and ele2!=None:
            r = Relation(internal_id,relation_type,ele1,ele2)

    if r!=None:
        return r


def getAllRelationsFromFile(relFile, csvFile):

    relations = []

    aspect_list = getAllAspectsFromFile(csvFile)
    subj_list = getAllSubjectivesFromFile(csvFile)

    f = open(relFile)

    for line in f.readlines():
        line = line.strip()

        r = getInfosFromLine(line,aspect_list,subj_list)
        if r!=None:
            relations.append(r)

    return relations

if __name__ == '__main__':

    relations = getAllRelationsFromFile("../files/en-coffeemachine-a1.rel","../files/en-coffeemachine-a1.csv")

    for r in relations:

        """
        if r.relation_type == "TARG-SUBJ":

            print r.internal_id
            print r.ele1.token
            print r.ele1.aspect_id
            print r.ele2.token
            print r.ele2.polar
            print
        """
        if r.relation_type == "COREF":
            print r.internal_id
            print r.ele1.token
            print r.ele1.aspect_id
            print r.ele2.token
            print r.ele2.aspect_id
            print