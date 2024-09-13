# Making Chatbot Arena Rating Calculation over 100x faster

The main code is in `faster.py` which contains faster implementations of the functions in the LMSYS leaderboard [notebook](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH).
`benchmark.py` and `parity_tests.ipynb` contain tests to confirm the same results are the same between the original and accelerated versions.

## How it works
The high level overview is:
* Avoid using for loops over pandas df, use numpy vectorization instead
* Exploit data duplication: Don't operate on the full `N` (1.6M) rows of the dataset, the likelihood can be computed using only the unique observed outcomes (22k) weighted by their occurance counts
* Exploit sparsity: the original uses logistic regression where input rows are vectors of dimension `M` (number of models) but all but 2 indices are 0. This version only stores and uses the indices of the current models
* Exploit symmetry: the gradient with respect to one model is the negative of the gradient with respect to the other, so you only need to compute one
* Do bootstrap sampling directly in the unique outcome space via the multinomial distribution, rather than sampling the full `N` with replacement
* Use multiprocessing to compute bootstrap samples all at once

I'll write up a more in detail blog post later
