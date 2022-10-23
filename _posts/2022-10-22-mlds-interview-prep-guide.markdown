---
layout: post
title:  "Machine Learning and Data Science - A Complete Interview Preperation Guide!"
date:   2022-10-22 00:00:01 -0700
categories: machine learning
---

After navigating the US Machine Learning and Data Science (MLDS) job market for a while, I wanted to share my journey, specific learning resources and tips to be better prepared and successful in these interviews. As the field is so dynamic at this point, the job titles, descriptions and interview processes can vary significantly. Nevertheless, the following guide is meant to be a holistic guide covering all aspects of the preparation.

 <h2>Table of Contents</h2>
- [Introduction](#introduction)
  - [Broad classification of MLDS job roles:](#broad-classification-of-mlds-job-roles)
- [Algorithmic Problem Solving](#algorithmic-problem-solving)
- [ML Coding](#ml-coding)
- [Machine Learning Theory](#machine-learning-theory)
  - [Grilling on Resume](#grilling-on-resume)
- [Specialty - Modern NLP, Search, Recommender Systems](#specialty---modern-nlp-search-recommender-systems)
- [ML Case Study and Design](#ml-case-study-and-design)
- [SQL](#sql)
- [Probability and Statistics](#probability-and-statistics)

## Introduction

Understanding the job role based on the company's profile and job description is the most important aspect to prioritize, prepare and apply for the kind of the positions you are interesed in. Although, jotting down all the duties of MLDS roles deserves a post in itself, I will broadly classify them into below three categories so that .

### Broad classification of MLDS job roles:
1. **Data Scientist, Applied ML Scientist, Applied ML Engineer** focusing on model research, model building and iterating, and working closely with ML infrastructure team to smoothly hand-off as packages, APIs etc.
2. **Software Engineer - ML, ML Infra Engineer, Data Engineer** focusing on building the data systems, and the backend stack that utilizes the real-time data points, ML model end points (sometimes treating them as blackbox) to serve the customers with minimal latency.
3. **Data Scientist, Business Analyst** focusing on Experiemental Design (A/B Testing), KPI design, dashboarding to track the impact across several dimensions, understand customer pain points and work closely with the Product owner to build the roadmap.

*Disclaimer: This classification is typical to a technology product company and may change for other industries like consulting etc. In some mid-size and startup companies, there are lot of 'Jack of all trades' roles that have good mix of the above roles*

Given these job roles, here are some pointers to keep in mind before reading the rest of the guide
* My preparation and experience is specific to mostly the first job classification above although there is good amount of overlap with the other two interms of preparation
* My experience is specific to the US job market which I believe is consistent with how global MLDS jobs and recruitment
* My primary focus areas are NLP, Ranking and Discovery systems (Search, Recommenders, Ad-Tech etc.) hence there is a seperate section on them

Given the primer, let's jump right into it!


## Algorithmic Problem Solving
The importance of Data Structures and Algorithms (DS&A) in judging a candidate's MLDS aptitude is a widely debatable topic. Although I personally hate these rounds, unfortunately, they are an easy adoption from software industry in eliminating false positives (are they really though?) from a wide pool of applicants. After going through numerous LeetCode lists and courses that promise to master DS&A in x months, here are the resources that helped me a lot in mainly providing a high level structure and timeline

* [**Tech Interview Handbook:**](https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/) This beautifully crafted free resource provides common patterns, corner cases, time complexity, essential questions, recommended practice questions for every Data structure and Algorithm which made it easy for me to master individual pieces before tackling them as a whole. [Here](https://www.techinterviewhandbook.org/algorithms/array/) is an example for Array data structure and it has covered all of them in detail.
* [**Neetcode 150:**](https://neetcode.io/practice) If you are practicing for coding interviews, you might probably already know him through his youtube channel. This resource combines commonly asked patterns and questions and was really helpful to revise/practice these questions
* [**Neetcode Video Solutions:**](https://www.youtube.com/c/NeetCode/playlists) A single best resource to understand the solutions. This can complement Leetcode Premium solutions and sometimes are even better than that. Note that the top-voted discussions on Leetcode also provide diverse ways solve the same problem.
* [**InterviewQuery:**](https://www.interviewquery.com/questions?searchQuery=&searchQuestionTag=&searchCompany=&ordering=Recommended&pageSize=20&page=0&tags=Python&tags=R&tags=Algorithms) I liked the algorithmic questions in this website (usually they are Easy/Medium level) and used them for quick practice when I felt I am overfitting my preparation for LeetCode. Although it has a premium plan, you can access most of the problems freely.
* **Python Tips and Tricks:** If you are using Python, which you should be (I honestly know only Python), you must utilize lot of non-common data structures which makes it easier and faster to implement your algorithm. These include data structures like `Counter`, `OrderedDict`, `defaultdict`, `heapq`, `deque` and many more. Leetcode discussion posts on 'Python one-liners' are actually a good place to take some notes on some useful stuff. Here are some links I referred while creating my tips:
  * [Python Tips and Tricks](https://aman.ai/code/python-tips/)
  * [useful python tricks for interviews and leetcode](https://leetcode.com/discuss/general-discussion/698708/useful-python-tricks-for-interviews-and-leetcode)
  * [Python-Interview-Tricks](https://github.com/amirgamil/Python-Interview-Tricks/blob/main/README.md)


Based on my interview experience, except for a few companies like TikTok, MLDS interviews focus on core and common DS&A. Honestly, I haven't solved a single problem on Trie or know the common bit-manipulation techniques. I would definitely recommend to prioritize in this order:
* **High Priority:** Arrays/Strings and Hashing, Two Pointers, Sliding Window
* **Medium Priroty:** Stack, Binary Search, 1-D/2D Dynamic Programming, Greedy, Intervals
* **Low Priority:** Trees, Heap, Backtracking, Graphs


## ML Coding
Along with traditional DS&A coding, some companies (like Walmart, C3.ai etc) will ask you to live code common ML algorithms like linear regression etc. without the use of any external libraries. The idea is to test your foundational knowledge of common ML patterns like forward pass, using the right loss function, vectorized gradient descent, stopping criteria, and evaluation metrics. This will also provide them an opportunity to check your design skills (classes, function I/O, naming patterns, test cases etc.) which are actually more representative of the actual work you will be doing. I would recommend to practice coding the following to be best prepared for this round:
* Linear Regression, Logistic Regression, K-Means Clustering
* Code the above from scratch using only NumPy or Torch (without AutoGrad)
* You should be able to accomodate interviewer requests on batch vs stochastic gradient descent, evaluating using R^2, RMSE, F1 Score etc.
* Code them in modular style. I found the `sklearn` APIs are easier to understand and simulate. For example, Linear Regression should be implemented using `model.fit(X, y)` and `model.predict(X)`

Here is an example on how you can implement a Linear Regression from scratch:

<details open>
<summary><b>Linear Regression Implementation from Scratch</b></summary>

{% highlight python %}
import numpy as np
import math


class LinearRegression:
    def __init__(self, epochs=100, lr=0.0001, bs=16):
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        
    def fit(self, X, y):
        m, n = X.shape
        iters = math.ceil(m/self.bs)
        self.loss = []
        
        # Initialize
        self.W = np.random.rand(n, 1)
        self.b = np.random.rand(1)
        
        # for each epoch and batch
        for epoch in range(self.epochs):
            
            print(f'Epoch: {epoch+1}')
            for i in range(iters):
                
                X_b = X[i*self.bs: i*self.bs + self.bs, :]
                y_hat = X_b @ self.W + self.b
                
                err = y[i*self.bs: i*self.bs + self.bs, :] - y_hat
                batch_loss = np.mean(err ** 2)
                self.loss.append(batch_loss)
                
                w_grad = -2/self.bs * X_b.T @ err
                b_grad = -2/self.bs * np.sum(err)
                
                self.W = self.W - self.lr * w_grad
                self.b = self.b - self.lr * b_grad
                
                print(f'Iter: {i}, Loss: {batch_loss}')
    
    def loss(self, y, y_hat):
        pass
    
    def predict(self, X):
        return X @ self.W + b
        

if __name__=='__main__':
    
    # Data
    np.random.seed(42)
    X = np.random.rand(100, 4)
    W = np.array([[0.2], [0.6], [-0.3], [0.1]])
    b = 2
    y = X @ W + b
    
    # Model
    model = LinearRegression(epochs=10000, lr=0.005, bs=32)
    model.fit(X, y)
    
    print(model.W, model.b)
{% endhighlight %}
</details>
<br>

## Machine Learning Theory
There will be definitely interview rounds focusing on Machine Learning Theory either as a standalone topic or in relation with your resume projects. Honestly, there is no one-stop solution that lists down everything you need for this round (and there can't be given the vast amount of topics), but here are the resources that helped me revise and stay confident on my ML theory.
* **Traditional ML Algorithms**: [Ace the Data Science Interview by Kevin Huo and Nick Singh](https://www.amazon.com/Ace-Data-Science-Interview-Questions/dp/0578973839/ref=sr_1_1?keywords=ace+the+data+science+interview&qid=1666492082&qu=eyJxc2MiOiIwLjk1IiwicXNhIjoiMC4zOCIsInFzcCI6IjAuMzAifQ%3D%3D&s=books&sprefix=ace+the+d%2Cstripbooks%2C242&sr=1-1) is definitely worth having for the condensed content they have put together. It's not a fully self-contained resource, but the XXX section and the practice questions helped me guide my theory exploration.
* **Deep Learning Theory**: If you already did a graduate level course in DL, use that as a reference. Otherwise, [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) and [Dive into Deep Learning](https://d2l.ai/) are wealth of free resources combining both theory and code across wide range of applications like Vision, NLP, Recommender Systems etc. 

If you want to mostly revise the content, I would recommend Googling 'different types of activations functions', 'pros and cons of different activation functions' and reading top 1-2 blogs and make your own notes. Use the below topics as a checklist:
* Different layers and their inductive biases (FC, convolution, pooling, RNN, LSTM, attention etc.) 
* Prominent model architectures: ResNet, Seq2Seq RNNs, Seq2Seq with Attention, Transformers, BERT
* Different types of loss functions and their utility
* Different types of activation functions and their motivations
* Different types of backprop algorithms and their motivations
* Strategies to reduce overfitting - Dropout, weight decay, Hyper parameter understanding etc.
* Vanishing and exploding gradients - why they occur and how to solve?
* Different learning paradigms: Pre-training and fine-tuning, self-supervised contrastive learning, semi-supervised learning etc.

[**Glassdoor's interview questions**](https://www.glassdoor.com/Interview/index.htm) really helpful to gauge the company's interview question bank and use them as practice to test your ML theory. Here are some collections I did:

<details>
<summary><b>Amazon Data Science Interview Questions</b></summary>
<!--All you need is a blank line-->

<li>How to decide k value in KNN</li>
<li>Explain unsupervised learning techniques which do not involve clustering</li>
<li>Diff techniques to assess multicollinearity (correlation, VIF, tolerance etc)</li>
<li>Explain random forest, discuss its pros and cons</li>
<li>name the five assumptions of linear regression</li>
<li>L1 L2 and why can L1 shrink weights to 0</li>
<li>how you will select an unbiased sample, deal with class imbalance, consider temporal effects. How will you split into train/val/test?</li>
<li>How decision tree built, overfitting etc.</li>
<li>What is p-value?</li>
<li>What is confidence Interval?</li>
<li>Assumptions of Linear regression</li>
<li>MSE vs MAE</li>
<li>How do you interpret logistic regression?</li>
<li>Code to calculate correlation of 2 vectors</li>
<li>p-value</li>
<li>tree-based modeling</li>
<li>boosting vs bagging</li>
<li>bias variance tradeoff</li>
<li>log reg and the descision boundry</li>
<li>gausian plane</li>
<li>normalization</li>
<li>random forest and types of trees.</li>
<li>cluster and regression</li>
<li>supervised and unsupervised learning</li>
<li>recommender system design</li>
<li>Math behind Principal Component Analysis</li>
<li>What is the difference between bagging and boosting?</li>
<li>Describe a case where how you have solved an ambiguous business problem using Machine Learning</li>
<li>difference between bagging and boosting</li>
<li>what is naive bayes</li>
<li>explain p-value in layman terms</li>
<li>what is a normal distribution.</li>
<li>Explain bias variance tradeoff</li>
<li>Example of a high bias and high variance models</li>
<li>How does dropout work?</li>
<li>What is L1 vs L2 regularization?</li>
<li>How would you improve a classification model that suffers from low precision?</li>
<li>Assumption of Linear Regression, etc.</li>
<li>What is over fitting</li>
<li>what to do with unbalanced data.</li>
<li>Why we have L1 and L2 regression regularizations but no L0.5 or L4?</li>
<li>questions about metric of classification problems</li>
<li>How do you calculate PCA</li>
<li>How to interpret the coefficient in Logistics Regression?</li>
<li>How statistical hypothesis testing works</li>
<li>Difference between linear regression and t-test</li>
<li>Explain in detail how a 1D CNN works.</li>
<li>Define loss function. Write the formula for the loss function</li>
<li>Explain SVM. Explain about C value in SVM</li>
<li>Having a categorical variable with thousands distinct values, how would you encode it?</li>
</details>
<br>

<details>
<summary><b>Walmart Data Science Interview Questions</b></summary>

<li>Describe how attention mechanism works in neural networks</li>
<li>Does R-square measure is sufficient for linear regression analysis. What if r-square value is low, does this mean that the fit is not good enough?</li>
<li>what is random forest?</li>
<li>given a dataset, what would be the output of the k-means algorithm?</li>
<li>what is a CNN?</li>
<li>Some detail questions on linear programing, convex optimization method. list out optimisation algorithms, How does simplex works? How do you handle a million variables in optimisation?
algorithms for integer programming</li>
<li>Linear Regression, Ridge, Lasso, Gradient Descent, PCA, k-means clustering</li>
<li>What is R-squared?</li>
<li>Inter-Cluster vs. Intra-Cluster distance measurements</li>
<li>How to handle missing data?</li>
<li>How to pick most important features in data?</li>
<li>Explain the BERT architecture and its advantage over a BiLSTM.</li>
<li>He asked a few questions about your projects, how to evaluate your results and then a few questions about decision tree and evaluation metrics.</li>
<li>Describe what is DBSCAN algorithm?</li>
<li>what is ROC curve?</li>
<li>what is confusion matrix?</li>
<li>Bagging and Boosting?</li>
<li>l2 norm closed form solution</li>
<li>derive loss function in MLP</li>
<li>Explain ROC curve, what does AUC represent?</li>
<li>What is F distribution?</li>
<li>What is cross entropy?</li>
<li>Difference between Gradient Boosting and Random Forest?</li>
<li>Assumptions of linear regression?</li>
<li>Significance of log odds?</li>
<li>Define parametric and non-parametric methods. Give some examples.</li>
<li>Generally, what happens to bias & variance as we increase the complexity of the model?</li>
<li>What is the intuition behind F1 score?</li>
<li>Explain Linear and Logistic Regression. List their assumptions. Why cannot we use Linear Regression on categorical output?</li>
<li>Explain Bias-Variance Tradeoff. Explain underfitting and overfitting. What is the need for regularization?</li>
<li>Explain variants of Gradient Descent and the pros and cons of each variant.</li>
<li>Difference between Bagging and Boosting. Explain Random Forest.</li>
<li>Explain Precision and Recall measures and give examples of use cases where each is measured.</li>
<li>Briefly discuss some dimension reduction techniques. Difference between PCA and SVD.</li>
<li>Explain ROC Curve. What do the axes of the ROC Curve represent? Elaborate on the two extreme points of the ROC Curve – (0, 0) and (1, 1).</li>
<li>Explain AUC and its physical interpretation? Is it possible to get AUC below 0.5? What is the worst AUC that you can possibly achieve?</li>
<li>Explain the problem of vanishing and exploding gradients. Briefly describe some methods to solve these.</li>
<li>What is the need for a pooling layer in CNNs? Difference between max pooling and average pooling.</li>
<li>Explain how will you forecast a time series? Can we perform Linear Regression on time-series data?</li>
<li>What is Central Limit Theorem? Give an example where it is used.</li>
<li>Why is hyperparameter tuning required? Elaborate on some common hyperparameters for tree-based models.</li>
<li>Briefly discuss some clustering methods. What are the drawbacks of K-Means Clustering?</li>
<li>Differentiate between generative and discriminative models and give examples of each.</li>

</details>
<br>

Note that the theory questions are tailored to your background, project experience and the job role at hand. For example, you are not expected to talk Reinforcement Learning if neither your background nor the role requires that. So, use the above resources accordingly.


### Grilling on Resume
Being prepared to answer what you did in your past projects is not enough. Most of interviewers asked questions on how I would approach past projects by changing the scenarios. Here are some scenarios to think about regarding your past projects:
* How do you think the performance will change if I instead use algorithm X in this project?
* How would you approach this problem instead if there were a) few labelled samples, b) no labelled samples?
* What would you consider if we put some constraints on latency, secondary metrics etc.?
* Can you think of strong no-ML baselines for this project?


## Specialty - Modern NLP, Search, Recommender Systems



## ML Case Study and Design



## SQL



## Probability and Statistics