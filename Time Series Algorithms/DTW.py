import numpy as np

def dtw(Sref,S):
    cost = np.zeros((len(Sref),len(S)))
    cost[0][0] = abs(Sref[0]-S[0])
    for i in xrange(1,len(Sref)):
        cost[i][0] = cost[i-1][0] + abs(Sref[i]-S[0])
    for j in xrange(1,len(S)):
        cost[0][j] = cost[0][j-1] + abs(Sref[0]-S[j])
    
    for i in xrange(1,len(Sref)):
        for j in xrange(1,len(S)):
            cost[i][j] = abs(Sref[i]-S[j]) + min(cost[i-1][j],cost[i-1][j-1],cost[i][j-1])

    return cost