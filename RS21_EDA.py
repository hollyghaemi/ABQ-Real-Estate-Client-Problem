#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:51:06 2019

@author: hallehghaemi
"""

#%% Basic dataframe analysis
df.head()
df.shape
list(df) #county_name is the variable we will attempt to predict
df.info()
df.describe()
df.corr()

#%% Output / Dependent Variable Analysis : county_name
# Print value counts

county_names_vector.unique()
print(county_names_vector.value_counts().count())
county_names_count =county_names_vector.value_counts()
county_names_count

# Create frequency distributions.
sns.set(style="darkgrid")
sns.barplot(county_names_count.index, county_names_count.values, alpha=0.9)
plt.title('Frequency Distribution of Counties')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Counties', fontsize=8)
#plt.show()

#Conclusion is that Bernalillo County (ABQ's county) is by far the most 
#populated county

#%% Categorical vs Continuous EDA

ax1 = sns.boxplot(x="county_name", y="population", data=df, palette="Set3").set_title("Population")
plt.xticks([0, 1, 2,3], ['Bernalillo','Sandoval', 'Torrance', 'Valencia'])
plt.show()
#county key:
#Bernalillo County 0
#Sandoval County 1
#Torrance County 2
#Valencia County 3
#Conclusion is that Bernallilo and Sandoval have normally distributed populations 
#whereas Torrance is highly skewed to the right and Valencia has a slight
#skew to the right as well which is likely a result of the fewer values available

ax2 = sns.boxplot(x="county_name", y="minority_population", data=df, palette="Set3").set_title("Minority Population")
plt.xticks([0, 1, 2,3], ['Bernalillo','Sandoval', 'Torrance', 'Valencia'])
plt.show()
#removing populations and all other variables that do not pertain to the client
#personally

ax3 = sns.boxplot(x="county_name", y="applicant_income_000s", data=df, palette="Set3").set_title("Applicant Income")
plt.xticks([0, 1, 2,3], ['Bernalillo','Sandoval', 'Torrance', 'Valencia'])
plt.show()
#applicant income has very obvious outliers with Bernalillo county having the greatest
#right skew

ax4 = sns.boxplot(x="county_name", y="loan_amount_000s", data=df, palette="Set3").set_title("Loan Amount")
plt.xticks([0, 1, 2,3], ['Bernalillo','Sandoval', 'Torrance', 'Valencia'])
plt.show()

#%% Continous EDA
# Make a separate data frame for continuous data.
cont_df = df.select_dtypes(include=['float64']).copy()
cont_df.head()
cont_df.describe()


#Correlation matrix of numerical values. Looks like there is 
g = sns.PairGrid(cont_df)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
plt.show()
#%% Categorical EDA
# Make a separate data frame for categorical data.
cat_df = df.select_dtypes(include=['int64']).copy()
cat_df.head()

# Print value counts and create frequency distributions.
print(cat_df['purchaser_type_name'].value_counts())
print(cat_df['purchaser_type_name'].value_counts().count())
carrier_count = cat_df['purchaser_type_name'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Purchase Types')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Purchase Types', fontsize=8)
plt.show()


print(cat_df['applicant_race_name_1'].value_counts())
print(cat_df['applicant_race_name_1'].value_counts().count())
carrier_count = cat_df['applicant_race_name_1'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Applicant Race')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('applicant_race_name_1', fontsize=8)
plt.show()