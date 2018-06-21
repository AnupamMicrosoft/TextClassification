# Classification Analysis on Textual Data

## Introduction
Statistical classification refers to the task of identifying a category, from a predefined set, to which a data point belongs, given a training data set with known category memberships. Classification differs from the task of clustering, which concerns grouping data points with no predefined category memberships, where the objective is to seek inherent structures in data with respect to suitable measures. Classification turns out as an essential element of data analysis, especially when dealing with a large amount of data. In this project, we look into different methods for classifying textual data. In this project, the goal includes: 
1. To learn how to construct tf-idf representations of textual data.   
2. To get familiar with various common classification methods.   
3. To learn ways to evaluate and diagnose classification results.   
4. To learn two dimensionality reduction methods: PCA & NMF.  
5. To get familiar with the complete pipeline of a textual data classification task.

## Getting familiar with the dataset
We work with “20 Newsgroups” dataset, which is a collection of approximately 20,000 documents, partitioned (nearly) evenly across 20 different newsgroups (newsgroups are discussion groups like forums, which originated during the early age of the Internet), each corresponding to a different topic. One can use fetch_20newsgroups provided by scikit-learn to load the dataset. Detailedusagescanbefoundathttp://scikit-learn.org/stable/datasets/twenty_ newsgroups.html In a classification problem one should make sure to properly handle any imbalance in the relativesizesofthedatasetscorrespondingtodifferentclasses. Todoso, onecaneithermodify the penalty function (i.e. assign more weight to errors from minority classes), or alternatively, down-sample the majority classes, to have the same number of instances as minority classes

### Task 1 - To get started, plot a histogram of the number of training documents per category to check if they are evenly distributed.
Note that the data set is already balanced (especially for the categories we’ll mainly work on) and so in this case we do not need to balance. But in general, as a data scientist you need to be aware of this issue.

## Binary Classification 
Togetstarted,weworkwithawellseparableportionofdata, andseeifwecantrainaclassifier that distinguishes two classes well. Concretely, let us take all the documents in the following classes:

Table 1: Two well-separated classes 
* Computer Technology -  comp.graphics comp.os.ms-windows.misc comp.sys.ibm.pc.hardware comp.sys.mac.hardware 
* Recreational Activity rec.autos rec.motorcycles rec.sport.baseball rec.sport.hockey  

Specifically, use the settings as the following code to load the data:   
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']   
train_dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)  
test_dataset = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)  

### Feature Extraction 
The primary step in classifying a corpus of text is choosing a proper document representation. A good representation should retain enough information that enable us to perform the classification, yet in the meantime, be concise to avoid computational intractability and over fitting. One common representation of documents is called “Bag of Words”, where a document is represented as a histogram of term frequencies, or other statistics of the terms, within a fixed vocabulary. As such, a corpus of text can be summarized into a term-document matrix whose entries are some statistic of the terms. First a common sense filtering is done to drop certain words or terms: to avoid unnecessarily large feature vectors (vocabulary size), terms that are too frequent in almost every document, or are very rare, are dropped out of the vocabulary. The same goes with special characters, common stop words (e.g. “and”, “the” etc.), In addition, appearances of words that share the same stem in the vocabulary (e.g. “goes” vs “going”) are merged into a single term. Further, one can consider using the normalized count of the vocabulary words in each document to build representation vectors. A popular numerical statistic to capture the importance of a word to a document in a corpus is the “Term Frequency-Inverse Document Frequency (TF-IDF)” metric. This measure takes into account count of the words in the document, as normalized by a certain function of the frequency of the individual words in the whole corpus. For example, if a corpus is about computer accessories then words such as “computer” “software” “purchase” will be present in almost every document and their frequency is not a distinguishing feature for any document in the corpus. The discriminating words will most likelybethosethatarespecializedtermsdescribingdifferenttypesofaccessoriesandhencewill occur in fewer documents. Thus, a human reading a particular document will usually ignore the contextually dominant words such as “computer”, “software” etc. and give more importance to specific words. This is like when going into a very bright room or looking at a bright object, thehumanperceptionsystemusuallyappliesasaturatingfunction(suchasalogarithm or square-root) to the actual input values before passing it on to the neurons. This makes sure that a contextually dominant signal does not overwhelm the decision-making processes in the brain. The TF-IDF functions draw their inspiration from such neuronal systems. Here we define the TF-IDF score to be 

tf-idf(d,t) = tf(t,d)×idf(t) 

wheretf(d,t) representsthefrequencyofterm t indocument d,andinversedocumentfrequency is defined as: 
   idf(t) = log( n df(t))+ 1 where n isthetotalnumberofdocuments,anddf(t) isthedocumentfrequency,i.e. thenumber of documents that contain the term t. 

### Task 2 - Use the following specs to extract features from the textual data: 
* Use the default stopwords of theCountVectorizer 
* Exclude terms that are numbers (e.g. “123”, “-45”, “6.7” etc.) 
* Performlemmatizationwithnltk.wordnet.WordNetLemmatizerandpos_tag 
* Usemin_df=3 ReporttheshapeoftheTF-IDFmatricesofthetrainandtestsubsetsrespectively

Please refer to the official documentation of CountVectorizer as well as the discussion section notebooks for details.

### Dimensionality Reduction 
After above operations, the dimensionality of the representation vectors (TF-IDF vectors) ranges in the order of thousands. However, learning algorithms may perform poorly in highdimensional data, which is sometimes referred to as “The Curse of Dimensionality”. Since the document-term TF-IDF matrix is sparse and low-rank, as a remedy, one can select a subset of the original features, which are more relevant with respect to certain performance measure, or transform the features into a lower dimensional space. In this project, we use two dimensionality reduction methods: Latent Semantic Indexing (LSI) and Non-negative Matrix Factorization (NMF), both of which minimize mean squared residual between the original data and a reconstruction from its low-dimensional approximation. Recall that our data is the term-document TF-IDF matrix, whose rows correspond to TF-IDF representation of the documents, i.e. 

#### LSI
The LSI representation is obtained by computing left and right singular vectors corresponding to the top k largest singular values of the term-document TF-IDF matrix X. We perform SVD to the matrix X, resulting in X = UΣVT, U and V orthogonal. Let the singular values in Σ be sorted in descending order, then the first k columns of U and V are called Uk and Vk respectively. Vk consists of the principle components in the feature space. Then we use (XVk) (which is also equal to (UkΣk)) as the dimension-reduced data matrix, where rows still correspond to documents, only that they can have (far) lower dimension. In this way, the number of features is reduced. LSI is similar to Principal Component Analysis (PCA), and you can see the lecture notes for their relationships. Having learnt U and V, to reduce the test data, we just multiply the test TF-IDF matrix Xt by Vk, i.e. Xt,reduced = XtVk. By doing so, we actually project the test TF-IDF vectors to the principle components, and use the projections as the dimension-reduced data.

#### NMF 
NMF tries to approximate the data matrix X∈Rn×m (i.e. we have n docs and m terms) with WH (W ∈ Rn×r, H ∈ Rr×m). Concretely, it finds the non-negative matrices W and H s.t. ∥X−WH∥2 F is minimized. Then we use W as the dim-reduced data matrix, and in the fit step, we calculate both W and H. The intuition behind this is that we are trying to describe the documents (the rows in X) as a (non-negative) linear combination of r topics: 


### Task 3-  Reduce the dimensionality of the data using the methods above 
* Apply LSI to the TF-IDF matrix corresponding to the 8 categories with k = 50; so each document is mapped to a 50-dimensional vector.   
* Also reduce dimensionality through NMF and compare with LSI: 

Which one is larger, the∥X−WH∥2 F in NMF or the X−UkΣkVT k 2 F in LSI? Why is the case?

#### Classification Algorithms

In this part, you are asked to use the dimension-reduced training data from LSI to train (different types of) classifiers, and evaluate the trained classifiers with test data. Your task would be to classify the documents into two classes “Computer Technology” vs “Recreational Activity”. Refer to Table 1 to find the 4 categories of documents comprising each of the two classes. In other words, you need to combine documents of those sub-classes of each class to form the set of documents for each class.

#### Classification Measures
Classification quality can be evaluated using different measures such as precision, recall, F-score, etc. Refer to the discussion material to find their definition.  
Depending on application, the true positive rate (TPR) and the false positive rate (FPR) have differentlevelsofsignificance. Inordertocharacterizethetrade-offbetweenthetwoquantities, we plot the receiver operating characteristic (ROC) curve. For binary classification, the curve is created by plotting the true positive rate against the false positive rate at various threshold settings on the probabilities assigned to each class (let us assume probability p for class 0 and 1−p for class 1). In particular, a threshold t is applied to value of p to select between the two classes. The value of threshold t is swept from 0 to 1, and a pair of TPR and FPR is got for each value of t. The ROC is the curve of TPR plotted against FPR.

#### SVM
Linear Support Vector Machines have been proved efficient when dealing with sparse high dimensional datasets, including textual data. They have been shown to have good generalization accuracy, while having low computational complexity. LinearSupportVectorMachinesaimtolearnavectoroffeatureweights,w,andanintercept, b, giventhetrainingdataset. Oncetheweightsarelearned,thelabelofadatapointisdetermined by thresholding wTx + b with 0, i.e. sign(wTx + b). Alternatively, one produce probabilities that the data point belongs to either class, by applying a logistic function instead of hard thresholding, i.e. calculating σ(wTx+ b). The learning process of the parameter w and b involves solving the following optimization problem

Minimizing the sum of the slack variables corresponds to minimizing the loss function on the trainingdata. Ontheotherhand,minimizingthefirstterm,whichisbasicallyaregularization, corresponds to maximizing the margin between the two classes. Note that in the objective function, each slack variable represents the amount of error that the classifier can tolerate for a given data sample. The tradeoff parameter γ controls relative importance of the two components of the objective function. For instance, when γ ≫ 1, misclassification of individual points is highly penalized, which is called “Hard Margin SVM”. In contrast, a “Soft Margin
5
SVM”,whichisthecasewhenγ ≪ 1,isverylenienttowardsmisclassificationofafewindividual points as long as most data points are well separated. 

### Task 4 - Hard margin and soft margin linear SVMs: • Train two linear SVMs and compare: – Train one SVM with γ = 1000 (hard margin), another with γ = 0.0001 (soft margin). – Plot the ROC curve, report the confusion matrix and calculate the accuracy, recall, precision and F-1 score of both SVM classiﬁer. Which one performs better? – What happens for the soft margin SVM? Why is the case? • Use cross-validation to choose γ: Using a 5-fold cross-validation, ﬁnd the best value of the parameter γ in the range {10k|− 3 ≤ k ≤ 3,k ∈ Z}. Again, plot the ROC curve and report the confusionmatrixandcalculatetheaccuracy,recallprecisionandF-1score of this best SVM

#### Logistic Regression
Although its name contains “regression”, logistic regression is a probability model that is used for binary classification. Inlogisticregression,alogisticfunction(σ(ϕ) = 1/(1+exp(−ϕ)))actingonalinearfunctionof thefeatures(ϕ(x) = wTx+b)isusedtocalculatetheprobabilitythatthedatapointbelongsto class 1, and during the training process, w and b that maximizes the likelihood of the training data are learnt. One can also add regularization term in the objective function, so that the goal of the training process is not only maximizing the likelihood, but also minimizing the regularization term, which is often some norm of the parameter vector w. Adding regularization helps prevent ill-conditioned results and over-fitting, and facilitate generalization ability of the classifier. A coefficient is used to control the trade-off between maximizing likelihood and minimizing the regularization term. 

### Task 5 - Logistic classiﬁer: • Trainalogisticclassiﬁer; plottheROCcurveandreporttheconfusionmatrix andcalculatetheaccuracy,recallprecisionandF-1scoreofthisclassiﬁer. • Regularization: – Using 5-fold cross-validation, ﬁnd the best regularization strength in the range{10k|−3 ≤ k ≤ 3,k ∈Z}forlogisticregressionwithL1regularization and logistic regression L2 regularization, respectively. – Compare the performance (accuracy, precision, recall and F-1 score) of 3 logistic classiﬁers: w/o regularization, w/ L1 regularization and w/ L2 regularization, using test data. How does the regularization parameter aﬀect the test error? How are the learnt coeﬃcients aﬀected? Why might one be interested in each type of regularization?

#### Naive Bayes
Scikit-learn provides a type of classifiers called “naïve Bayes classifiers”. They includeMultinomialNB, BernoulliNB, and GaussianNB. Naïve Bayes classifiers use the assumption that features are statistically independent of each other when conditioned by the class the data point belongs to, to simplify the calculation for the Maximum A Posteriori (MAP) estimation of the labels. That is,
P(xi |y,x1,...,xi−1,xi+1,...,xm) = P(xi |y) i ∈{1,...,m} where xi’s are features, i.e. components of a data point, and y is the label of the data point. Now that we have this assumption, a probabilistic model is still needed; the difference between MultinomialNB, BernoulliNB, and GaussianNB is that they use different models. 

### Task 6 -Naïve Bayes classiﬁer: train a GaussianNB classiﬁer; plot the ROC curve and report the confusion matrix and calculate the accuracy, recall precisionandF-1scoreof this classiﬁer

### Grid Search of Parameters
Now we have gone through the complete process of training and testing a classifier. However, there are lots of parameters that we can tune. In this part, we fine-tune the parameters. 

### Task 7 -  Grid search of parameters: • ConstructaPipelinethatperformsfeatureextraction,dimensionalityreduction and classiﬁcation; • Dogridsearchwith5-foldcross-validationtocomparethefollowing(usetest accuracy as the score to compare):What is the best combination?

## Multiclass Classification
So far, we have been dealing with classifying the data points into two classes. In this part, we explore multiclass classification techniques through different algorithms. Someclassifiersperformthemulticlassclassificationinherently. Assuch,naïveBayesalgorithm finds the class with maximum likelihood given the data, regardless of the number of classes. In fact, the probability of each class label is computed in the usual way, then the class with the highest probability is picked; that is.
For SVM, however, one needs to extend the binary classification techniques when there are multiple classes. A natural way to do so is to perform a one versus one classification on all (|C| 2)pairs of classes, and given a document the class is assigned with the majority vote. In case there is more than one class with the highest vote, the class with the highest total classification confidence levels in the binary classifiers is picked. An alternative strategy would be to fit one classifier per class, which reduces the number of classifiers to be learnt to |C|. For each classifier, the class is fitted against all the other classes. Note that in this case, the unbalanced number of documents in each class should be handled. By learning a single classifier for each class, one can get insights on the interpretation of the classes based on the features. 

### Task 8 - Inthispart,weaimtolearnclassiﬁersonthedocumentsbelonging to the classes: 
comp.sys.ibm.pc.hardware,comp.sys.mac.hardware, 
misc.forsale,soc.religion.christian 
Perform Naïve Bayes classiﬁcation and multiclass SVM classiﬁcation (with both One VS One and One VS the rest methods described above) and report theconfusion matrix and calculate the accuracy, recall, precision and F-1 score of your classiﬁers.




