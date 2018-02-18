from average_precision import *
import re
import sys

p = re.compile('[^a-zA-Z0-9]')

def ngramRank(orderName,sentTotal):

    counts = []

    bigList = []

    for i in range(sentTotal):
        bigList.append([])

    for i in range(101)[1:]:
        for j in range(sentTotal):
            fileName = orderName+"/fm_filter_sizes_1_"+str(i)
            lines = list(open(fileName, "r").readlines())
            bigList[j].append(lines[j])

    #print len(bigList)

    for bg in bigList:
        count = {}
        sentence = ""
        for sent in bg:
            eles = sent.split('\t\t')
            predict_label = eles[1]
            gold_label = eles[0]
            ngram = eles[2]
            sentence = eles[3]
            if not count.has_key(ngram):
                count[ngram] = 1
            else:
                count[ngram] += 1
        count = sorted(count.items(), key=lambda x:x[1], reverse=True)

        #print count
        input = [ele[1] for ele in count]
        #print input
        noDup = list(set(input))
        noDup.sort(reverse=True)
        #print noDup
        rankNgram = []
        for ele in count:
            rankNgram.append((ele[0].lower(),noDup.index(ele[1])))

        #print rankNgram

        #counts.append(["".join(sentence.lower().strip().split()),predict_label,gold_label,rankNgram])
        counts.append([re.sub(p,'',sentence.lower().strip()),predict_label,gold_label,rankNgram])

    return counts

#def writeDownNgram(orderName):
def sortNgram(orderName,polar,sentTotal):

    polarNgram = []

    counts = ngramRank(orderName,sentTotal)

    counts.sort()
    for sent in counts:
        gold_label = sent[2]
        if gold_label==polar:
            polarNgram.append(sent)

    #print len(polarNgram)
    #for ele in polarNgram:
        #print ele

    return polarNgram

def subjAspeGold(fileName):

    #fw = open('classifier1c/withSubj-pos-test-sorted-a1.txt', "w")
    #lines = list(open('classifier1c/withSubj-pos-test-a1.txt', "r").readlines())
    lines = list(open(fileName, "r").readlines())
    newLines = []

    for line in lines:
        eles = line.split('\t')
        #newLines.append("".join(eles[0].lower().split())+'\t'+eles[1]+'\t'+eles[2]+'\t'+eles[3])
        newLines.append(re.sub(p,'',eles[0].lower())+'\t'+eles[1]+'\t'+eles[2]+'\t'+eles[3])

    newLines.sort()
    return newLines

def subjAspeGoldFull(fileName,orderName,prod):
    list_gold = list(open(fileName,"r").readlines())
    list_test = list(open(orderName+'/fm_filter_sizes_1_1','r').readlines())

    #print len(list_gold)
    #print len(list_test)

    list_gold_test = []

    for line1 in list_gold:
        eles = line1.split('\t')
        sent1 = re.sub(p,'',eles[0].lower())
        nornalSent = eles[0]
        for line2 in list_test:
            eles2 = line2.split('\t\t')
            sent2 = re.sub(p,'',eles2[3].lower())
            gold_label = eles2[0]
            if sent1.startswith(sent2[:30]) and gold_label==prod:
                list_gold_test.append(sent1+'\t'+eles[1]+'\t'+eles[2]+'\t'+eles[3]+'\t'+nornalSent)

    #print len(list_gold_test)

    list_gold_test.sort()

    #print len(list_gold_test)

    #for ele in list_gold_test:
        #print ele
    return list_gold_test

def makeRankingListProd(ngramOrder,testGold,flag,sentTotal):

    #polarNgram = sortNgram("feature_maps_test_tuned_sentiment","neg")
    #goldSubjAspe = subjAspeGold('classifier1c/withSubj-neg-test-a1.txt')

    polarNgram = sortNgram(ngramOrder,flag,sentTotal)
    #goldSubjAspe = subjAspeGold(testGold)
    goldSubjAspe = subjAspeGoldFull(testGold,ngramOrder,flag)

    #print len(polarNgram)
    #print len(goldSubjAspe)


    goldList = []
    ngramList = []

    for polarNgram,goldSubjAspe in zip(polarNgram,goldSubjAspe):

        #print polarNgram
        #print goldSubjAspe

        gold = []
        aspe = goldSubjAspe.split('\t')[3].strip()
        gold.append((aspe.lower(),0))
        #print gold

        ngram = polarNgram[3]
        #print ngram
        goldList.append(gold)
        ngramList.append(ngram)

    return goldList,ngramList

def makeRankingListPolarFirst2(ngramOrder,testGold,flag,sentTotal):

    #polarNgram = sortNgram("feature_maps_test_tuned_sentiment","neg")
    #goldSubjAspe = subjAspeGold('classifier1c/withSubj-neg-test-a1.txt')

    polarNgram = sortNgram(ngramOrder,flag,sentTotal)
    #goldSubjAspe = subjAspeGold(testGold)
    goldSubjAspe = subjAspeGoldFull(testGold,ngramOrder,flag)

    #print len(polarNgram)
    #print len(goldSubjAspe)


    goldList = []
    ngramList = []

    for polarNgram,goldSubjAspe in zip(polarNgram,goldSubjAspe):

        #print polarNgram
        #print goldSubjAspe

        gold = []
        subj = goldSubjAspe.split('\t')[2]
        gold.append((subj.lower(),0))
        #print gold

        ngram = polarNgram[3]
        #print ngram
        goldList.append(gold)
        ngramList.append(ngram)

    return goldList,ngramList

def makeRankingList(ngramOrder,testGold,flag,sentTotal):

    #polarNgram = sortNgram("feature_maps_test_tuned_sentiment","neg")
    #goldSubjAspe = subjAspeGold('classifier1c/withSubj-neg-test-a1.txt')

    polarNgram = sortNgram(ngramOrder,flag,sentTotal)
    goldSubjAspe = subjAspeGold(testGold)

    #print len(polarNgram)
    #print len(goldSubjAspe)


    goldList = []
    ngramList = []

    for polarNgram,goldSubjAspe in zip(polarNgram,goldSubjAspe):

        #print polarNgram
        #print goldSubjAspe

        gold = []
        subj = goldSubjAspe.split('\t')[2]
        gold.append((subj.lower(),0))
        #print gold

        ngram = polarNgram[3]
        #print ngram
        goldList.append(gold)
        ngramList.append(ngram)

    return goldList,ngramList

if __name__ == '__main__':

    #ngramRank("feature_maps_test_tuned")

    #writeDownNgram("feature_maps_test_tuned")

    #subjAspeGold('classifier1c/withSubj-pos-test-a1.txt')


    goldList_pos,ngramList_pos = makeRankingList('feature_maps_test_'+sys.argv[1]+'_sentiment','classifier1c/withSubj-pos-test-a1.txt','pos',220)
    goldList_neg,ngramList_neg = makeRankingList('feature_maps_test_'+sys.argv[1]+'_sentiment','classifier1c/withSubj-neg-test-a1.txt','neg',220)

    goldList = goldList_pos+goldList_neg
    ngramList = ngramList_pos+ngramList_neg


    print "mean average precison, "+sys.argv[1]+": " + str(mapk_tie(goldList,ngramList))


    #subjAspeGoldFull('sents/cutlery-withAsp-a1.txt','feature_maps_test_tuned_product','cutlery')

    #sortNgram('feature_maps_test_tuned_product','cutlery',310)


    #goldList,ngramList = makeRankingListProd('feature_maps_test_tuned_product','sents/coffeemachine-withAsp-a1.txt','coffeemachine',310)

    #print mapk_tie(goldList,ngramList)


    #goldList,ngramList = makeRankingListPolarFirst2('feature_maps_test_tuned_sentiment','sents/withSubj-pos-a1.txt','pos',220)
    #print mapk_tie(goldList,ngramList)


