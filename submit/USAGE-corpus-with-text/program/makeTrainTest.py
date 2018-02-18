import random

def makeTrainTestData():

    # positive: 1483, negative: 718
    # positive-test: 148, negative-test: 72

    #positive_sents = list(open("../sents/classifier1/relCorReplaced-pos-a1.txt", "r").readlines())
    #negative_sents = list(open("../sents/classifier1/relCorReplaced-neg-a1.txt", "r").readlines())

    positive_sents = list(open("../sents/withSubj-pos-a1.txt", "r").readlines())
    negative_sents = list(open("../sents/withSubj-neg-a1.txt", "r").readlines())

    positive_para = list(open("../paraphrases/subjective-pos-a1.txt","r").readlines())
    negative_para = list(open("../paraphrases/subjective-neg-a1.txt","r").readlines())

    random.seed(10)
    random.shuffle(positive_sents)
    random.shuffle(negative_sents)
    random.shuffle(positive_para)
    random.shuffle(negative_para)

    positive_train = positive_sents[148:]
    positive_test = positive_sents[:148]

    negative_train = negative_sents[72:]
    negative_test = negative_sents[:72]

    positive_para_unseen = []
    negative_para_unseen = []

    for para in positive_para:
        unseen = 'n'
        for sent in positive_test:
            if para.strip() in sent:
                unseen = 'y'
        for sent in negative_test:
            if para.strip() in sent:
                unseen = 'y'
        if unseen == 'n':
            positive_para_unseen.append(para)

    #print len(positive_para)
    #print len(positive_para_unseen)

    for para in negative_para:
        unseen = 'n'
        for sent in negative_test:
            if para.strip() in sent:
                unseen = 'y'
        for sent in positive_test:
            if para.strip() in sent:
                unseen = 'y'
        if unseen == 'n':
            negative_para_unseen.append(para)

    #print len(negative_para)
    #print len(negative_para_unseen)

    test_sent_with_subj_pos = []

    for sent in positive_test:
        subj = set([])
        for para in positive_para:
            if para.strip() in sent:
                subj.add(para.strip()+'_'+"pos")
        for para in negative_para:
            if para.strip() in sent:
                subj.add(para.strip()+'_'+"neg")
        test_sent_with_subj_pos.append(sent.strip()+"\t"+str(subj))

    test_sent_with_subj_neg = []

    for sent in negative_test:
        subj = set([])
        for para in positive_para:
            if para.strip() in sent:
                subj.add(para.strip()+'_'+"pos")
        for para in negative_para:
            if para.strip() in sent:
                subj.add(para.strip()+'_'+"neg")
        test_sent_with_subj_neg.append(sent.strip()+"\t"+str(subj))


    return positive_train, positive_test, negative_train, negative_test, positive_para_unseen, negative_para_unseen, test_sent_with_subj_pos, test_sent_with_subj_neg



def writeDownSentences():

    data = makeTrainTestData()
    fw1 = open('../sents/classifier1b/relCorRep-pos-train-a1.txt','w')
    fw2 = open('../sents/classifier1b/relCorRep-pos-test-a1.txt','w')
    fw3 = open('../sents/classifier1b/relCorRep-neg-train-a1.txt','w')
    fw4 = open('../sents/classifier1b/relCorRep-neg-test-a1.txt','w')
    fw5 = open('../sents/classifier1b/subjective-pos-a1.txt','w')
    fw6 = open('../sents/classifier1b/subjective-neg-a1.txt','w')

    for sent in data[0]:
        fw1.write(sent)

    for sent in data[1]:
        fw2.write(sent)

    for sent in data[2]:
        fw3.write(sent)

    for sent in data[3]:
        fw4.write(sent)

    for para in data[4]:
        fw5.write(para)

    for para in data[5]:
        fw6.write(para)

def testSubjExtract():

    data = makeTrainTestData()

    for sent in data[7]:
        print sent

def writeDownSentencesWithSubjAndAspe():

    data = makeTrainTestData()
    fw2 = open('../sents/withSubj-pos-test-a1.txt','w')
    fw4 = open('../sents/withSubj-neg-test-a1.txt','w')

    for sent in data[1]:
        fw2.write(sent)

    for sent in data[3]:
        fw4.write(sent)


if __name__ == '__main__':

    #writeDownSentences()
    #testSubjExtract()

    writeDownSentencesWithSubjAndAspe()