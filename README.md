# MachineLearning-Naive_Bayes

## Naive Bayes

1. Naïve Bayes is a classification technique based on Bayes’ Theorem.
2. Let’s assume that you are data scientist working major bank in NYC and you want to classify a new client as eligible to retire or not.
3. Customer features are his/her age and salary.

### PRIOR PROBABILITY

1. Points can be classified as RED or BLUE. 
2. Our task is to classify a new point to RED or BLUE.
3. Prior Probability: Since we have more BLUE compared to RED, we can assume that our new point is twice as likely to be BLUE than RED. 

𝑃𝑟𝑖𝑜𝑟 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 𝑓𝑜𝑟 𝑅𝐸𝐷=(𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑅𝐸𝐷 𝑃𝑜𝑖𝑛𝑡𝑠)/(𝑇𝑜𝑡𝑎𝑙 𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑃𝑜𝑖𝑛𝑡𝑠)=20/60

𝑃𝑟𝑖𝑜𝑟 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 𝑓𝑜𝑟 𝐵𝐿𝑈𝐸=(𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝐵𝐿𝑈𝐸 𝑃𝑜𝑖𝑛𝑡𝑠)/(𝑇𝑜𝑡𝑎𝑙 𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑃𝑜𝑖𝑛𝑡𝑠)=40/60

### LIKELIHOOD

1. For the new point, if there are more BLUE points in its vicinity, it is more likely that the new point will be classified as BLUE. 
2. So we draw a circle around the point
3. Then we calculate the number of points in the circle belonging to each class label.

𝐿𝑖𝑘𝑒𝑙𝑖ℎ𝑜𝑜𝑑 𝑜𝑓 𝑋 𝑏𝑒𝑖𝑛𝑔 𝑅𝐸𝐷=(𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑅𝐸𝐷 𝑃𝑜𝑖𝑛𝑡𝑠 𝑖𝑛 𝑣𝑖𝑐𝑖𝑛𝑖𝑡𝑦 )/(𝑇𝑜𝑡𝑎𝑙 𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑅𝐸𝐷 𝑃𝑜𝑖𝑛𝑡𝑠)=3/20

𝐿𝑖𝑘𝑒𝑙𝑖ℎ𝑜𝑜𝑑 𝑜𝑓 𝑋 𝑏𝑒𝑖𝑛𝑔 𝐵𝐿𝑈𝐸=(𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝐵𝐿𝑈𝐸 𝑃𝑜𝑖𝑛𝑡𝑠 𝑖𝑛 𝑣𝑖𝑐𝑖𝑛𝑖𝑡𝑦 )/(𝑇𝑜𝑡𝑎𝑙 𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝐵𝐿𝑈𝐸 𝑃𝑜𝑖𝑛𝑡𝑠)=1/40

### POSTERIOR PROBABILITY

1. Let’s combine prior probability and likelihood to create a posterior probability. 
2. Prior probabilities: suggests that X may be classified as BLUE Because there are twice as much blue points.
3. Likelihood: suggests that X is RED because there are more RED points in the vicinity of X.
4. Bayes’ Rule combines both to form a posterior probability.

𝑃𝑜𝑠𝑡𝑒𝑟𝑖𝑜𝑟 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 𝑜𝑓 𝑋 𝑏𝑒𝑖𝑛𝑔 𝑅𝐸𝐷= 𝑃𝑟𝑖𝑜𝑟 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 𝑜𝑓 𝑅𝐸𝐷 ∗𝐿𝑖𝑘𝑒𝑙𝑖ℎ𝑜𝑜𝑑 𝑜𝑓 𝑋 𝑏𝑒𝑖𝑛𝑔 𝑅𝐸𝐷=20/60∗3/20=1/20  

𝑃𝑜𝑠𝑡𝑒𝑟𝑖𝑜𝑟 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 𝑜𝑓 𝑋 𝑏𝑒𝑖𝑛𝑔 𝐵𝐿𝑈𝐸= 𝑃𝑟𝑖𝑜𝑟 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦 𝑜𝑓 𝐵𝐿𝑈𝐸∗𝐿𝑖𝑘𝑒𝑙𝑖ℎ𝑜𝑜𝑑 𝑜𝑓 𝑋 𝑏𝑒𝑖𝑛𝑔 𝐵𝐿𝑈𝐸=40/60∗1/40=1/60  

## To Get Clear Understanding, Refer the Power Points
