#!/usr/bin/python
# -*- coding: utf-8 -*-
from relation import *

class Element:

    def __init__(self, internal_id, product_id, review_id, product, title, reviewText, disappeared, subjectives, aspects, relations):
        self.internal_id = internal_id
        self.product_id = product_id
        self.review_id = review_id
        self.product = product
        self.title = title
        self.reviewText = reviewText
        self.disappeared = disappeared
        self.subjectives = subjectives
        self.aspects = aspects
        self.relations = relations

def getInfosFromLine(line, rels_list, aspects_list, subjs_list):
    eles = line.split('\t')
    length = len(eles)
    relations = []
    aspects = []
    subjectives = []

    internal_id = eles[0]
    for rel in rels_list:
        if rel.internal_id == internal_id:
            relations.append(rel)

    for aspect in aspects_list:
        if aspect.internal_id == internal_id:
            aspects.append(aspect)

    for subj in subjs_list:
        if subj.internal_id == internal_id:
            subjectives.append(subj)

    if length==4:
        e = Element(internal_id, eles[1], eles[2], "", "", "", "y", subjectives, aspects, relations)
    elif length==6:
        e = Element(internal_id, eles[1], eles[2], eles[3], eles[4], eles[5], "n", subjectives, aspects, relations)

    return e


def getAllElementsFromFile(txtFile, relFile, csvFile):

    eles  = []

    rels_list = getAllRelationsFromFile(relFile,csvFile)
    aspects_list = getAllAspectsFromFile(csvFile)
    subjs_list = getAllSubjectivesFromFile(csvFile)

    f = open(txtFile)
    for line in f.readlines():
        line = line.strip()
        e = getInfosFromLine(line, rels_list, aspects_list, subjs_list)
        eles.append(e)

    return eles

if __name__=='__main__':

    eles = getAllElementsFromFile("../files/en-coffeemachine.txt", "../files/en-coffeemachine-a1.rel","../files/en-coffeemachine-a1.csv")

    for e in eles:

        #print "Element:"


        print "title: " +e.title
        print "product id: "+e.product_id
        print "review text: "+e.reviewText
        #print "disappeared: "+e.disappeared

        #print

        for r in e.relations:

            print "relations:"

            if r.relation_type == "TARG-SUBJ":

                print "internal_id: "+r.internal_id
                print "aspect token: "+r.ele1.token
                print "aspect id: " +r.ele1.aspect_id
                print "subjective token: "+r.ele2.token
                print "subjective polar: "+r.ele2.polar
                print

        #for s in e.subjectives:
            #print "subjective id: "+s.subj_id
            #print "subjetive token: "+s.token
            #print "subjective polar: "+s.polar
            #print
