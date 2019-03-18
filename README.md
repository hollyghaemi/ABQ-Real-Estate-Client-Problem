# RS21
RS21 ABQ Real Estate Client Problem

With data sets regarding ABQ public transportation (2012), twitter tweets, CDC health measures (2017), facebook check-ins and
home mortgage applications (2014) -- data clean up, exploratory data analysis and model selection led to the following 
conclusions.

The client is assumed to be a real estate agency that would like to speed up their process of matching customers with 
counties. The home mortgage applications data set was the basis for running algorithms with the task of classification. 3 
algorithms were run: logistic regression, random forest and multi-layer perceptron. It was found that the logistic regression
model had the highest accuracy at 81%. This means that the model allows the real estate agency to use its client information 
to classify the county that they will most likely want to live in correctly, 81% of the time. 

The ABQ public transportation, facebook check-ins, twitter tweets and CDC health measures were all similar in the fact that 
they had associated geolocations. These geolocations were uploaded into ArcGis to easily create maps with information
on the various locations of health issues, popular restaurants and bus stops. The real estate agency will use these maps on 
their first interaction with their client to show them the ins and outs of the county that they previously classified their
client with. 
