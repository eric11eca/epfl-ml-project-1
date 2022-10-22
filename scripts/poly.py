def build_poly(x, degree, *args, pretreated=True):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Return the augmented basis matrix tx.
    If pretreated = True, we suppose the first 4 columns are cathegorical (they sum up to the identity).
    """

    if pretreated:
        n,p = x.shape
        tx = np.zeros((n,(degree*(p-4))+4))
        tx[:,:4] = x[:,:4]
        for feature in range(p-4):
            for i in range(1,degree+1):
                tx[:,3+feature*degree+i]=(x[:,4+feature])**i
        return tx
    else:
        n,p = x.shape
        tx = np.zeros((n,(degree*p)+1))
        tx[:,0] = 1
        for feature in range(p):
            for i in range(1,degree+1):
                tx[:,feature*degree+i]=(x[:,feature])**i
        return tx