# CardiovascularComplicationPrediction
Project for "CS4641 - Machine Learning" by Team 46: Aditya Kumar, Farouk Marhaba, Kinnera Banda, and Maya Rajan.

![summary figure](./images/randomBanner.jpg)

## Introduction/Background
Coronary heart disease (CHD), the most common type of heart disease, kills over 300,000 people in the United States annually. It is caused by a buildup of plaque in the arteries that supply blood to the heart, limiting blood flow and increasing the risk of heart attacks. However, with simple preventative measures, actions can be taken to significantly reduce the risk of CHD early on. This includes, but is not limited to, decreasing and eliminating smoking habits, eating foods higher in fiber and lower in saturated fat, and becoming more active. 

Our project aims to predict the 10-year risk of coronary heart disease by observing factors about an individual’s characteristics, including health metrics and lifestyle choices. With accurate predictions on whether or not an individual is at risk for CHD over the next 10 years, we can advise at-risk individuals to make lifestyle changes in the present to minimize the risk of CHD in the future. Early risk assessment is critical in the fight against coronary heart disease, and our team is excited to address this.

## Methods
* Describe the dataset
    * 4,000 records and 15 attributes
    * Columns
        * Nominal
            * Current Smoker: whether or not the patient is a current smoker 
            * BP Meds: whether or not the patient was on blood pressure medication 
            * Prevalent Stroke: whether or not the patient had previously had a stroke 
            * Prevalent Hyp: whether or not the patient was hypertensive 
            * Diabetes: whether or not the patient had diabetes 
        * Continuous
            * Tot Chol: total cholesterol level
            * Sys BP: systolic blood pressure
            * Dia BP: diastolic blood pressure 
            * BMI: Body Mass Index
            * Heart Rate: heart rate
            * Glucose: glucose level 
            * Cigs Per Day: the number of cigarettes that the person smoked on average in one day
        * Medical history
        * Predict variable (desired target)
        * Output
            * 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)
* Possible Techniques
    * Supervised learning
        * Classification: tries to estimate mapping function from input to discrete output variables
    * Unsupervised learning
        * Finding relationships between features 
        * Principal Component Analysis for feature reduction


## Results
Our team recognizes the importance of an early prognosis of cardiovascular diseases. By showing the factors that most closely predict the onset of coronary heart disease (CHD), we hope to aid others, especially patients at high-risk, in making lifestyle decisions that may reduce future complications. With our data from Framingham, Massachusetts, we plan to employ classification to discern the traits most closely associated with CHD and the traits most closely associated with no disease. We also plan to employ linear regression to ensure that these traits can make accurate predictions on CHD development. Using these models, we hope to create a robust prediction that can be employed with new data from other cities.

## Discussion
With the methods specified above, CHD rates in particular areas and understands which features weight more than the others. With these results, policy changes and recommendations can be made. For example, determining if the number of hospitals or the insurance policy or cost has a direct effect on CHD risk. The purpose would be to analyze which combination of features affect the risk of CHD of a person living that area the most. Reproducing these results on different cardiovascular datasets would prove the effectiveness of our model. A potential next step would be to extend our algorithm to predicting not just heart disease risk but also the likelihood of other illnesses and analyze the effect of different combinations of features on the risk. For example, breast cancer, cervical cancer, etc. 

## References
Dalen, James et. al. “The Epidemic of the 20th Century: Coronary Heart Disease.” The American Journal of Medicine, 2014. Retrieved 25 September 2020 from https://www.amjmed.com/article/S0002-9343(14)00354-4/pdf

“Coronary Artery Disease: Prevention, Treatment and Research.” Johns Hopkins Medicine. Retrieved 25 September 2020 from https://www.hopkinsmedicine.org/health/conditions-and-diseases/coronary-artery-disease-prevention-treatment-and-research

“Heart Disease Facts.” Centers for Disease Control and Prevention, 2020. Retrieved 25 September 2020 from https://www.cdc.gov/heartdisease/facts.htm

