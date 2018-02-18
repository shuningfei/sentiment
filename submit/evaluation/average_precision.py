import numpy as np

def apk_tie(actual, predicted, k=10):

    #print actual
    #print predicted

    # actual: e.g. [('a',0)]
    # predicted e.g. [('m',0),('n',1),('a',1),('t',2)]

    # actual [('great',0)]
    # predicated [('great for when ',0),('marvelous feature is ',1),("do n't want ", 2),('clear or smoke ', 3), ("n't want to ", 4), ('sitting on my ', 4)]

    score = 0.0
    num_hits = 0.0

    real_actual = []

    for ele in actual:
        real_actual.append(ele[0])

    for j in range(len(predicted)):
        i = predicted[j][1]
        p = predicted[j][0]

        #print "i: "+str(i)
        #print "p: "+str(p)

        temp = predicted[:j]
        pre = []
        for t in temp:
            pre.append(t[0])

        #print "pre: " +str(pre)

        #if p in real_actual and p not in pre:

        allMatch = False

        if p not in pre:

            match = False
            for ele in real_actual:
                gold=ele.split(" ")
                gram = p.split(" ")

                for e1 in gold:
                    for e2 in gram:
                        if e1==e2:
                            match = True
                            allMatch = True
            if match==True:

                num_hits += 1.0
                score += num_hits / (i+1.0)

                #print "num_hits: " + str(num_hits)
                #print "score: " + str(score)

        if allMatch:
            break

    if not actual:
        return 0.0

    #print "end score: " + str(score / min(len(actual), k))
    return score / min(len(actual), k)

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    #print len(predicted)

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0


    for i,p in enumerate(predicted):

        print "i: "+str(i)
        print "p: "+str(p)

        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

            print "num_hits: " + str(num_hits)
            print "score: " + str(score)

    if not actual:
        return 0.0

    #print "end score: " + str(score / min(len(actual), k))
    return score / min(len(actual), k)

def mapk_tie(actual, predicted, k=10):

    return np.mean([apk_tie(a,p,k) for a,p in zip(actual, predicted)])

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

if __name__=='__main__':
    #apk(['a'],['m','n','a','t'])  # 0.333
    #apk(['a'],['m','a','t'])   # 0.5
    #print mapk([[1],['a']],[[2,3,4],['a','l','n']])

    #apk_tie([('a',0)],[('m',0),('n',1),('a',1),('t',2)]) # 0.5
    #apk_tie([('a',0)],[('m',0),('a',1),('n',1),('t',2)]) # 0.5
    #apk_tie([('a',0)],[('m',0),('a',1),('a',1),('t',2)]) # 0.5
    apk_tie([('t',0)],[('m',0),('a',1),('a',1),('t',2)])

    #apk_tie([('great',0)],[('great for when ',0),('marvelous feature is ',1),("do n't want ", 2),('clear or smoke ', 3), ("n't want to ", 4), ('sitting on my ', 4)])

    #print mapk_tie([[('a',0)],[('great',0)]],[[('m',0),('a',1),('a',1),('t',2)],[('great for when ',0),('marvelous feature is ',1),("do n't want ", 2),('clear or smoke ', 3), ("n't want to ", 4), ('sitting on my ', 4)]])