def retainedcal(S,K):
    
    ret = 1.0*S[:K].sum() /S.sum()
    print 'when K=%d,variance retained:%0.4f' %(K, ret)
    return ret