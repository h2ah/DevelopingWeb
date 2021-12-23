def computeDistance(s,t):
    cache = {} # (m,n)=>  result
    def recurse(m,n):
        """0
        Return the minimum edit distance between:
        - first m letters of s
        - first n letters of t
        """
        if (m,n) in cache:
            return cache[(m,n)]
        if m == 0:  # Base case
            result = n # Base case
        elif s[m - 1] == t[n - 1]:
            result = recurse(m - 1,n - 1)
        else:
            subCost = 1 + recurse(m-1, n-1)
            delCost = 1 + recurse(m-1,n)
            insCost = 1 + recurse(m, n-1)
            result = min(subCost, delCost, insCost)
        cache[(m,n)] = result
        return result

    return recurse(len(s),len(t))
print(computeDistance('a cat'*20,'the cats!'*20))
