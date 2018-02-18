#!/usr/bin/python
# -*- coding: utf-8 -*-

from element import *
import nltk.data

def changeReviewToSentences(txtFile,relFile,csvFile,lang):

    d = buildCorefAspectDict(relFile,csvFile)

    all_sents = []

    subj_sents = []

    aspe_sents = []

    subj_aspe_sents = []

    rel_sents = []

    #rel_sents_polar = []

    nothing_sents = []

    only_subj_sents = []

    only_aspe_sents = []

    subj_aspe_notRel_sents = []

    not_subj_sents = []

    not_aspe_sents = []

    not_rel_sents = []

    sentsCorefRep = []

    sentsRelCorefRep = []

    sentsRelCorefRep_polar = []

    if lang == 'en':
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    elif lang == 'de':
        sent_detector = nltk.data.load('tokenizers/punkt/german.pickle')

    #eles = getAllElementsFromFile("../files/en-coffeemachine.txt", "../files/en-coffeemachine-a1.rel","../files/en-coffeemachine-a1.csv")
    eles = getAllElementsFromFile(txtFile,relFile,csvFile)

    for e in eles:

        title_coref = 'n'

        for asp in d:

            if e.internal_id == asp.internal_id:

                begin_posi_orig = asp.begin_posi
                end_posi_orig = asp.end_posi

                #if e.title[int(begin_posi_orig):int(end_posi_orig)] != None:
                coref = e.title[int(begin_posi_orig):int(end_posi_orig)]
                if coref == asp.token:
                    #print asp.internal_id
                    #print e.title
                    #print coref
                    token_after = e.title[:int(begin_posi_orig)]+d[asp].token+e.title[int(end_posi_orig):]
                    #print token_after
                    sentsCorefRep.append(token_after)
                    title_coref = 'y'
                    #print

        subj_begin_posi_s = []
        aspe_begin_posi_s = []
        rel_begin_posi_s_polar = []

        #polarRel = []

        for subj in e.subjectives:
            subj_begin_posi_s.append(subj.begin_posi)

        for aspe in e.aspects:
            aspe_begin_posi_s.append(aspe.begin_posi)

        for rela in e.relations:

            if rela.relation_type == 'TARG-SUBJ':
                rel_begin_posi_s_polar.append((rela.ele1.begin_posi,rela.ele2.begin_posi,rela.ele2.polar,rela.ele2.token,rela.ele1.token)) # ele1 -> apse; ele2 -> subj

        title_subj = 'n'
        title_aspe = 'n'
        title_rel = 'n'

        title_polar = ""

        for pos in range(len(e.title)):

            if str(pos) in subj_begin_posi_s:
                title_subj = 'y'

            if str(pos) in aspe_begin_posi_s:
                title_aspe = 'y'

        for pos1 in range(len(e.title)):
            for pos2 in range(len(e.title)):
                for relEle in rel_begin_posi_s_polar:
                    if str(pos1)==relEle[0] and str(pos2)==relEle[1]:
                        #if (str(pos1),str(pos2)) in rel_begin_posi_s:
                        title_rel = 'y'
                        title_polar = relEle[2] + "\t" + relEle[3] + "\t" + relEle[4]


        if title_subj == 'y':
            subj_sents.append(e.title)

        if title_aspe == 'y':
            aspe_sents.append(e.title)

        if title_subj == 'y' and title_aspe == 'y':
            subj_aspe_sents.append(title_subj)

        if title_rel == 'y' :
            rel_sents.append(e.title)

        if title_subj == 'n' and title_aspe == 'n':
            nothing_sents.append(e.title)

        if title_subj == 'y' and title_aspe == 'n':
            only_subj_sents.append(e.title)

        if title_aspe == 'y' and title_subj == 'n':
            only_aspe_sents.append(e.title)

        if title_subj == 'y' and title_aspe == 'y' and title_rel == 'n':
            subj_aspe_notRel_sents.append(e.title)

        if title_subj == 'n':
            not_subj_sents.append(e.title)

        if title_aspe == 'n':
            not_aspe_sents.append(e.title)

        if title_rel == 'n':
            not_rel_sents.append(e.title)

        if title_rel == 'y':

            for asp in d:

                if e.internal_id == asp.internal_id:

                    begin_posi_orig = asp.begin_posi
                    end_posi_orig = asp.end_posi

                    #if e.title[int(begin_posi_orig):int(end_posi_orig)] != None:
                    coref = e.title[int(begin_posi_orig):int(end_posi_orig)]
                    if coref == asp.token:
                        #print asp.internal_id
                        #print e.title
                        #print coref
                        token_after = e.title[:int(begin_posi_orig)]+d[asp].token+e.title[int(end_posi_orig):]
                        #print token_after

                        sentsRelCorefRep.append(token_after)
                        sentsRelCorefRep_polar.append(title_polar)
                        title_coref = 'y'
                        #print

        if title_rel == 'y' and title_coref == 'n':
            sentsRelCorefRep.append(e.title)
            sentsRelCorefRep_polar.append(title_polar)

        all_sents.append(e.title)

        sentences = sent_detector.tokenize(e.reviewText.strip())

        currentPos = len(e.title)

        for sent in sentences:

            sent_coref = 'n'

            sent_subj = 'n'
            sent_aspe = 'n'
            sent_rel = 'n'

            sent_polar = 'y'

            all_sents.append(sent)

            for pos in range(currentPos+len(sent))[currentPos:]:

                if str(pos) in subj_begin_posi_s:
                    sent_subj = 'y'

                if str(pos) in aspe_begin_posi_s:
                    sent_aspe = 'y'

                for asp in d:

                    if pos == int(asp.begin_posi) and e.internal_id == asp.internal_id:

                        begin_posi_orig = asp.begin_posi
                        end_posi_orig = asp.end_posi

                        coref = sent[int(begin_posi_orig)-currentPos-1:int(end_posi_orig)-currentPos-1]
                        if coref == asp.token:

                            #print asp.aspect_id
                            #print sent
                            #print coref
                            token_after = sent[:int(begin_posi_orig)-currentPos-1]+d[asp].token+sent[int(end_posi_orig)-currentPos-1:]
                            #print token_after
                            #print
                            sent_coref = 'y'
                            sentsCorefRep.append(token_after)

            for pos1 in range(currentPos+len(sent))[currentPos:]:
                for pos2 in range(currentPos+len(sent))[currentPos:]:
                    for relEle in rel_begin_posi_s_polar:
                        if str(pos1) == relEle[0] and str(pos2) == relEle[1]:
                            sent_rel = 'y'
                            sent_polar = relEle[2] + "\t" +relEle[3] + "\t" + relEle[4]

            if sent_subj=='y':
                subj_sents.append(sent)

            if sent_aspe=='y':
                aspe_sents.append(sent)

            if sent_subj == 'y' and sent_aspe == 'y':
                subj_aspe_sents.append(title_subj)

            if sent_rel=='y':
                rel_sents.append(sent)

            if sent_subj=='n' and sent_aspe=='n':
                nothing_sents.append(sent)

            if sent_subj == 'y' and sent_aspe == 'n':
                only_subj_sents.append(sent)

            if sent_aspe == 'y' and sent_subj == 'n':
                only_aspe_sents.append(sent)

            if sent_subj == 'y' and sent_aspe == 'y' and sent_rel == 'n':
                subj_aspe_notRel_sents.append(sent)

            if sent_subj == 'n':
                not_subj_sents.append(sent)

            if sent_aspe == 'n':
                not_aspe_sents.append(sent)

            if sent_rel == 'n':
                not_rel_sents.append(sent)

            if sent_rel == 'y':

                for pos in range(currentPos+len(sent))[currentPos:]:

                    for asp in d:

                        if pos == int(asp.begin_posi) and e.internal_id == asp.internal_id:


                            begin_posi_orig = asp.begin_posi
                            end_posi_orig = asp.end_posi

                            coref = sent[int(begin_posi_orig)-currentPos-1:int(end_posi_orig)-currentPos-1]
                            if coref == asp.token:

                                #print asp.aspect_id
                                #print sent
                                #print coref
                                token_after = sent[:int(begin_posi_orig)-currentPos-1]+d[asp].token+sent[int(end_posi_orig)-currentPos-1:]
                                #print token_after
                                #print

                                #print token_after

                                sentsRelCorefRep.append(token_after)
                                sentsRelCorefRep_polar.append(sent_polar)

            if sent_rel == 'y' and sent_coref == 'n':
                sentsRelCorefRep.append(sent)
                sentsRelCorefRep_polar.append(sent_polar)

            currentPos = currentPos+len(sent)+1

    return all_sents,subj_sents,aspe_sents,subj_aspe_sents,rel_sents,nothing_sents,only_subj_sents,only_aspe_sents,subj_aspe_notRel_sents,not_subj_sents,not_aspe_sents,sentsCorefRep,sentsRelCorefRep,sentsRelCorefRep_polar,not_rel_sents


def writeSentences():

    products = ['coffeemachine','cutlery','microwave','toaster','trashcan','vacuum','washer']

    #onemore = 'dishwasher'

    lang = ['en','de']

    anno = ['a1','a2']


    for an in anno:

        #fw1 = open('../sents/relCorReplaced-pos-'+an+'.txt','w')
        #fw2 = open('../sents/relCorReplaced-neg-'+an+'.txt','w')

        #fw1 = open('../sents/withSubj-pos-'+an+'.txt','w')
        #fw2 = open('../sents/withSubj-neg-'+an+'.txt','w')

        #fw3 = open('../sents/relCorReplaced-neu-'+an+'.txt','w')
        #fw_only_subj = open('../sents/only_subj_sents-'+an+'.txt','w')
        #fw_only_aspe = open('../sents/only_aspe_sents-'+an+'.txt','w')
        #fw_rel = open('../sents/rel_sents-'+an+'.txt','w')
        #fw_nothing = open('../sents/nothing_sents-'+an+'.txt','w')
        #fw_subj = open('../sents/subj_sents-'+an+'.txt','w')
        #fw_not_subj = open('../sents/not_subj_sents-'+an+'.txt','w')
        #fw_aspe = open('../sents/aspe_sents-'+an+'.txt','w')
        #fw_not_aspe =  open('../sents/not_aspe_sents-'+an+'.txt','w')
        #fw_rel = open('../sents/rel_sents-'+an+'.txt','w')
        #fw_not_rel =  open('../sents/not_rel_sents-'+an+'.txt','w')

        for prod in products:

            #fw_p = open('../sents/'+prod+'-'+an+'.txt','w')
            fw_p = open('../sents/'+prod+'-withAsp-'+an+'.txt','w')

            #for la in lang:
            sents = changeReviewToSentences('../files/'+'en-'+prod+'.txt','../files/'+'en-'+prod+'-'+an+'.rel','../files/'+'en-'+prod+'-'+an+'.csv','en')

            #sents = changeReviewToSentences(txtFile,relFile,csvFile)

            all_sents = sents[0]

            subj_sents = sents[1]

            aspe_sents = sents[2]

            subj_aspe_sents = sents[3]

            rel_sents = sents[4]

            nothing_sents = sents[5]

            only_subj_sents = sents[6]

            only_aspe_sents = sents[7]

            subj_aspe_notRel_sents = sents[8]

            not_subj_sents = sents[9]

            not_aspe_sents = sents[10]

            sentsCorefRep = sents[11]

            sentsRelCorefRep = sents[12]

            sentsRelCorefRep_polar = sents[13]

            not_rel_sents = sents[14]

            """
            for pair in zip(sentsRelCorefRep,sentsRelCorefRep_polar):


                if pair[1].startswith('neutral'):
                    fw3.write(pair[0]+'\n')


                if pair[1].startswith('positive'):
                    fw1.write(pair[0]+'\t'+pair[1]+'\n')
                elif pair[1].startswith('negative'):
                    fw2.write(pair[0]+'\t'+pair[1]+'\n')
            """
            """
            for sent in only_subj_sents:
                fw_only_subj.write(sent+'\n')

            for sent in only_aspe_sents:
                fw_only_aspe.write(sent+'\n')

            for sent in rel_sents:
                fw_rel.write(sent+'\n')

            for sent in nothing_sents:
                fw_nothing.write(sent+'\n')
            """

            #for sent in sentsRelCorefRep:

                #fw1.write(sent+'\n')


            """
            for sent in subj_sents:
                fw_subj.write(sent+'\n')

            for sent in not_subj_sents:
                fw_not_subj.write(sent+'\n')

            for sent in aspe_sents:
                fw_aspe.write(sent+'\n')

            for sent in not_aspe_sents:
                fw_not_aspe.write(sent+'\n')

            for sent in rel_sents:
                fw_rel.write(sent+'\n')

            for sent in not_rel_sents:
                fw_not_rel.write(sent+'\n')
            """

            #for sent in sentsRelCorefRep:
                #fw_p.write(sent+'\n')

            for pair in zip(sentsRelCorefRep,sentsRelCorefRep_polar):
                fw_p.write(pair[0]+'\t'+pair[1]+'\n')
"""
def writeSentences():



    products = ['coffeemachine','cutlery','microwave','toaster','trashcan','vacuum','washer']

    anno = ['a1','a2']

    for pro in products:
        for an in anno:
            fw1 = open('../relCorReplaced/'+pro+'/-'+an+'.txt','w')
            fw1.write()
"""

if __name__=='__main__':

    #classSentences("../files/en-coffeemachine.txt", "../files/en-coffeemachine-a1.rel","../files/en-coffeemachine-a1.csv")
    writeSentences()

    #changeReviewToSentences("../files/en-coffeemachine.txt", "../files/en-coffeemachine-a1.rel","../files/en-coffeemachine-a1.csv",'en')
