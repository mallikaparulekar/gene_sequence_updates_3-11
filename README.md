# gene_sequence_updates_3-11
This contains updates to gene_seq version2, including more accurate cross validation code, with binary search optimization
Description of the files:
crossvalSlow: contains updated code for cross validation used to find the optimal number of clusters for a given data set, and has higher accuracy than previous versions

crossvalBinSums: Chooses the direction to continue the calculation in based on the weightedProbability sum for each cluster. For example, if the the three middle cluster values have probabilities in ascending order, it would continue to the right side (greater than)

CrossvalWay2: same idea as BinSums, except it uses maximums from each cross val to account for outliers, instead of just taking one sum.
