1. Predicting. We have two options I believe. One, predict pairwise sims using NN treating it as a binary classification model. Second, use metric learning and add a l2 distance metric function at the end of network. Then, convert distance to pairwise sim and output. What is really the difference between these approaches? IN this case, we use metric learning methods to train. I.e., we can use queried edges to determine triples to select (for triple loss).

2. Inferring. We can infer information based on queried edges. However, is this worth the effort compared to simply using the feature vectors in order to quickly imply information, similar to COBRAS. 



