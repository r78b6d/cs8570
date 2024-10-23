java cAssignment 2 (DDL: 2024/10/20) 
 
1. Point Estimation (15 pts) 
The Poisson distribution is a useful discrete distribution which can be used to model the 
number of occurrences of something per unit time. For example, in networking, packet arrival 
density is often modeled with the Poisson distribution. If   is Poisson distributed, i.e., its probability mass function takes the following form: 
 
2. Source of Error: Part 1 (15 pts) 
Suppose that we are given an independent and identically distributed sample of   points { } 
where each point   ∼  ( , 1) is distributed according to a normal distribution with mean   
and variance 1. You are going to analyze different estimators of the mean  . 
(a) Suppose that we use the estimator  ̂= 1 for the mean of the sample, ignoring the 
observed data when making our estimate. Give the bias and variance of this estimator  ̂. 
Explainin a sentence whether this is a good estimator in general, and give an example of 
when this is a good estimator. 
(b) Now suppose that we use  ̂=  $ as an estimator of the mean. That is, we use the first 
data point in our sample to estimate the mean of the sample. Give the bias and variance 
of thisestimator  ̂. Explain in a sentence or two whether this is a good estimator or not. 
(c) In the class you have seen the relationship between the MLE estimator and the least 
squares problem. Sometimes it is useful to use the following estimate 
'$
 
For the mean, where the parameter   > 0 is a known number. The estimator  ̂ is biased, 
but has lower variance than the sample mean  ̅=  "$ ∑   which is an unbiased 
estimator for  . Give the bias and variance of the estimator  ̂. 
 
3. Source of Error: Part 2 (15 pts) 
In class we discussed the fact that machine learning algorithms for function approximation 
are also a kind of estimator (of the unknown target function), and that errors in function 
approximation arise from three sources: bias, variance, and unavoidable error. In this part of 
the question you are going to analyze error when training Bayesian classifiers. Suppose that   is boolean,   is real valued,  (  = 1) = 1/2 and that the class conditional 
distributions  ( | ) are uniform distributions with  ( |  = 1) =        [1,4] and 
 ( |  = 0) =        [−4, −1]. (we use        [ ,  ] to denote a uniform probability 
distribution between   and  , with zero probability outside the interval [ ,  ]). 
(a) Plot the two class conditional probability distributions  ( |  = 0) and  ( |  = 1). 
(b) What is the error of the optimal classifier? Note that the optimal classifier knows  (  =
1) ,  ( |  = 0) and  ( |  = 1) perfectly, and applies Bayes rule to classify new 
examples. Recall that the error of a classifier is the probability that it will misclassify a new 
  drawn at random from  ( ). The error of this optimal Bayes classifier is the unavoidable 
error for this learning task. 
(c) Suppose instead that  (  = 1) = 1/2 and that the class conditional distributions are 
uniform distribution with  ( |  = 1) =        [0,4] and  ( |  = 0) =
       [−3,1]. What isthe unavoidable error in this case? Justify your answer. 
(d) Consider again the learning task from part (a) above. Suppose we train a Gaussian Naive 
Bayes (GNB) classifier using   training examples for this task, where   → ∞. Of course our 
classifier will now (incorrectly) model  ( | ) as a Gaussian distribution, so it will be 
biased: it cannot even represent the correct form of  ( | ) or  ( | ). 
Draw again the plot you created in part (a), and add to it a sketch of the learned/estimated 
class conditional probability distributions the classifier will derive from the infinite training 
data. Write down an expression for the error of the GNB. (hint: your expression will 
involve integrals - please don't bother solving them). 
(e) So far we have assumed infinite training data, so the only two sources of error are bias 
and unavoidable error. Explain in one sentences how your answer to p代 写program、c/c++，Python
代做程序编程语言art (d) above would 
change if the number of training examples was finite. Will the error increase or decrease? 
Which of the three possible sources of error would be present in this situation? 
 
4. Gaussian (Naïve) Bayes and Logistic Regression (15 pts) 
Recall that a generative classifier estimates  ( ,  ) = 	 ( ) ( | ), while a discriminative 
classifier directly estimates  ( | ). (Note that certain discriminative classifiers are nonprobabilistic:
they directly estimate a function  ∶   →   instead of  ( | ).) For clarity, we 
highlight   in bold to emphasize that it usually represents a vector of multiple attributes, i.e., 
  = { $,  +, . . . ,  %}. However, this question does not require students to derivethe answer 
in vector/matrix notation. 
In class we have observed an interesting relationship between a discriminative classifier 
(logistic regression) and a generative classifier (Gaussian naive Bayes): the form of 
 ( | )	derived from the assumptions of a specific class of Gaussian naive Bayes classifiers is 
precisely the form used by logistic regression. The derivation can be found in the required 
reading: http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf.We made the following 
assumptions for Gaussian naive Bayes classifiers to model  ( ,  ) =  ( ) ( | ): 
(1)   is a boolean variable following a Bernouli distribution, with parameter   =  (  = 	1) 
and thus  (  = 0) = 1 −  . 
(2)   = { $,  +, . . . ,  %}, where each attribute   is a continuous random variable. For each 
  ,  ( |  =  ) is a Gaussian distribution  ( ,,  ) . Note that   is the standard 
deviation of the Gaussian distribution (and thus  
+ is the variance), which does not 
depend on  . 
(3) For all   ≠  ,   and  - are conditionally independent given  . This is why this type of 
classifier is called “naive”. We say this is a specific class of Gaussian naive Bayes classifiers because we have made an 
assumption that the standard deviation   of  ( |  =  ) does not depend on the value   of 
 . This is not a general assumption for Gaussian naive Bayes classifiers. 
Let's make our Gaussian naive Bayes classifiers a little more general by removing the 
assumption that the standard deviation   of  ( |  =  ) does not depend on  . As a result, 
for each  ,  ( |  =  ) is Gaussian distribution  ( ,,  ,), where   = 1,2, . . . ,   and   =
0,1. Note that now the standard deviation  , of  ( |  =  ) depends on both the attribute 
index   and the value   of  . 
Question: is the new form of  ( | ) implied by this more general Gaussian naive Bayes 
classifier still the form used by logistic regression? Derive the new form of  ( | ) to prove 
your answer. 
 
5. Programming (40 pts) 
In this lab, please submit your code according to the following guidelines: 
(a) Cross-Validation: https://qffc.uic.edu.cn/home/content/index/pid/276/cid/6530.html 
Please try these three approaches holdout, K-fold and leave-p-out with the data file 2.1-
Exercise.csv. 
Submit ‘Exercise-handout.py’, ‘Exercise-k-fold.py’, and ‘Exercise-leave-p-out.py’ 
(b) Linear regression: https://qffc.uic.edu.cn/home/content/index/pid/276/cid/6541.html 
Please modify linear_regression_lobf.py with the data file 2.2-Exercise.csv. For this task, 
take the High column values as variables and Target column for prediction. 
Submit ‘Exercise-linear_regression_lobf.py’ 
(c) Naïve Bayes: https://qffc.uic.edu.cn/home/content/index/pid/276/cid/6557.html 
Here the dataset ‘basketball.csv’ used is for basketball games and weather conditions 
where the target is if a basketball game is played in the given conditions or not, the 
dataset is very small, just containing 14 rows and 5 columns. 
Submit ‘Exercise-NB.py’ 
(d) Logistic regression: https://qffc.uic.edu.cn/home/content/index/pid/276/cid/6556.html 
Use breast cancer from sklearn using following code: from sklearn.datasets import 
load_breast_cancer. 
Submit ‘Exercise-Logistic-Regression.py’ 

         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
