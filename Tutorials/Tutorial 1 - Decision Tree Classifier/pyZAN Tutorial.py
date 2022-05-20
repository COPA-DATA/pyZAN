# Python Mini-Tutorial - Decision Tree Classifier
#
# For questions please contact: support@copadata.com
#
# In this tutorial we're going to:
#   1) read zenon archive data into a pandas dataframe using pyZAN
#   2) visualize the data using seaborn
#   3) make some simple calculations and statistics with that data
#   4) bring the data into a good shape for our ml algorithm
#   5) train a decision tree to classify our data into "healthy" and "unhealthy"
#   6) load some more data from zenon to verify that our ml model works

# The tutorial is based on the zenon project PREDICTIVE_MAINTENANCE_DEMO_820.
# In that project a simple simulation creates cyclic welding data.
# To follow this tutorial you will need:
#   - a zenon supervisor > V 8.20 to run the project
#   - a zenon analyzer > V 3.40  set up with a filled meta db for the project
#   - pyZAN installed (pip install CopaData)

# We will focus on data from Roboter 1 (R1). In order to repeat the results from this
# tutorial you will need to generate a set of healthy an unhealthy data. 
# 30 minutes for each set should be fine. Unhealthy data can be created by 
# activating one of the simulated errors in the zenon project. In this case I'm
# working with the simplest error by decreasing welding current (Error 3)

from CopaData import pyZAN
import datetime
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics



# ------------------------------------------------------------
# ------------------ Part 1 - Read the data ------------------
# ------------------------------------------------------------

# First connect to our analyzer
zan = pyZAN.Server(server='localhost', metadb='ZA_Predictive820')

#Test
# Get projects, archives and variables
projects = zan.read_MetaData_Projects()
archives = zan.read_MetaData_Archives()
variables = zan.read_MetaData_Variables()
                                                        
# We will focus on R1_WeldingCurrent, R1_WeldingResistance, R1_WeldingVoltage and R1_Status
# Let's get our healthy data

healthy = zan.read_Online_Archive(project_reference = "PREDICTIVE_MAINTENANCE_DEMO_820",\
                                 archive_reference = "PA",\
                                 variable_references = [\
                                 "RobotSimForPA/Global/R1_WeldingResistance",\
                                 "RobotSimForPA/Global/R1_WeldingCurrent",\
                                 "RobotSimForPA/Global/R1_Status",\
                                 "RobotSimForPA/Global/R1_WeldingVoltage"],\
                                 time_from = datetime.datetime(2019,12,3,7,20,0),\
                                 time_to = datetime.datetime(2019,12,3,7,50,0),\
                                 show_visualnames=True)




# in case you want to use my data uncomment this line
#healthy = pd.read_csv("healthy.csv",index_col=0, parse_dates=True)

healthy.set_index("TIMESTAMP",drop=True,inplace = True)
# healthy.set_index(["TIMESTAMP","VARIABLE_NAME"],drop=True,inplace = True)


# Our zenon data has a lot of really usefull columns like STATUSFLAGS and UNIT
# For our simple purposes we don't need them... we'll discard all columns but VALUE
healthy = healthy[['VARIABLE_NAME','VALUE']]

# The column VARIABLE_NAME has quite longish values... make them shorter
healthy['VARIABLE_NAME'] = healthy['VARIABLE_NAME'].apply(lambda x: x.replace('RobotSimForPA/Global/',''))
#healthy = healthy.sort_values(by='TIMESTAMP')


# We will pivot our dataframe to give us one column for each variable
healthy = healthy.reset_index().pivot(index='TIMESTAMP', columns='VARIABLE_NAME',values='VALUE')

# ------------------------------------------------------------
# ------------------ Part 2 - Visualize the data -------------
# ------------------------------------------------------------


# lets have a look
fig, axs = plt.subplots(2)
healthy.plot(y="R1_WeldingCurrent",ax=axs[0],legend=False)
healthy.plot(y="R1_WeldingVoltage",ax=axs[0],legend=False)
ax2 = axs[0].twinx()
healthy.plot(y="R1_Status", ax=ax2,color="r",legend=False)
axs[0].set_title('current, voltage and status')
healthy.plot(y='R1_WeldingResistance',ax=axs[1],legend=False)
axs[1].set_title('resistance')
fig.suptitle('healthy data')
plt.show()

# ------------------------------------------------------------
# ------------------ Part 3 - Simple statistics --------------
# ------------------------------------------------------------

# Looking at these plots it is quite obvious that there is a high correlation
# between current,voltage and resistance. Not too surprising... the simulation 
# uses ohms law to calculate one from the other
# To quantify this we'll have look at the cross correlation matrix
corr = healthy.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr,cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask)
fig.suptitle('cross correlation matrix')
plt.tight_layout()

# Another observation is, that our data has a cyclic nature with a cycle of
# roughly 30 seconds. Let's calculate the autocorrelation of R1_WeldingCurrent.
def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm,mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = acorr.argmax()+1
    r = acorr[lag-1]        
    if np.abs(r) > 0.5:
      print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    else: 
      print('Appears to be not autocorrelated')
    return acorr,r, lag

acorr,r, lag = autocorr(healthy.R1_WeldingCurrent)

fig, ax = plt.subplots()
plt.plot(acorr)
fig.suptitle('auto correlation')
# For my test-data we have r=0.89 for lag=31... so our best guess for the length
# of a single cycle is 31 seconds
# We want to analyze all those "Welding Cycles" and therefore need their data
# split into parts, where each part contains a single cycle.





# ------------------------------------------------------------
# ----------- Part 4 - Bring data in a good shape ------------
# ------------------------------------------------------------

# A closer look reveals, that the cylce length is not constant
# and has a slight variation of a few seconds.
# So we can't just slice the data after every 31 seconds.
# But obviously R1_Status can serve as a marker for the beginning of each cycle
# since it switches from 1 to 0 at each beginning cycle.
# We will use that to slice our healthy data into pieces

def sliceData(x,lag):
    sliced = pd.DataFrame(columns = ['nr','time_offset'] + x.columns.tolist())
    sliced.set_index(['nr','time_offset'],inplace=True)
    
    flex=lag//8
    i=2*flex
    count=0
    offset = list(range(0,lag))
    while i < len(x.index)-lag:
        # find a switch of R1_Status to 0
        if x.iloc[i]['R1_Status']==0 and x.iloc[i-1]['R1_Status']!=0:
            sl = pd.DataFrame(data=x[i-2*flex:i-2*flex+lag])
            sl['time_offset'] = offset
            sl.set_index(['time_offset'],inplace=True)
            sliced = sliced.append(sl.assign(nr=count).set_index('nr',append=True).swaplevel(0,1))
            count = count + 1
            i=i+lag-flex
        else:
            i=i+1
            
    return sliced
    
healthy_sliced = sliceData(healthy,lag)

# Let's plot the first 10 slices
fig,ax = plt.subplots()
for i in range(0,10):
    sns.lineplot(x=healthy_sliced.loc[(i,),].index,y=healthy_sliced.loc[(i,),'R1_WeldingCurrent'],ax=ax)   
fig.suptitle('healthy data - sliced')

# Get a mean slice ...
healthy_mean_slice = healthy_sliced.mean(level=1)
fig,ax = plt.subplots()
plt.plot(healthy_mean_slice.R1_WeldingCurrent,label='R1_WeldingCurrent')
fig.legend()
fig.suptitle('healthy data - mean slice')



# ------------------------------------------------------------
# ------------------- Part 5 - Train ML model ----------------
# ------------------------------------------------------------


# -------------------------- ML Fun begins here -----------------------------

# We are going for one of the simplest ML algorithm to start with: decision tree
# For now we don't care about the theory behind it. There is enough info on
# the internets for this... To make it short:
# We give our decision tree training data (our welding slices from above) and we
# tell him whether a specific curve is from "healthy" oder "unhealthy" welding.
# After the trainig we can present a welding data slice to the trained model
# and it will tell us, whether it is "healthy" or "unhealthy"

# First of all we will transform our healthy_sliced dataframe...
# We will focus on R1_WeldingCurrent and throw away anything else
healthy_sliced = healthy_sliced.R1_WeldingCurrent

# now we pivot that data, giving us one row for each slice with 31 (lag) columns
healthy_ml = healthy_sliced.unstack()

# our healthy data is ready... we now need an unhealthy data set
unhealthy = zan.read_Online_Archive(project_reference = "PREDICTIVE_MAINTENANCE_DEMO_820",\
                                 archive_reference = "PA",\
                                 variable_references = [\
                                 "RobotSimForPA/Global/R1_WeldingResistance",\
                                 "RobotSimForPA/Global/R1_WeldingCurrent",\
                                 "RobotSimForPA/Global/R1_Status",\
                                 "RobotSimForPA/Global/R1_WeldingVoltage"],\
                                 time_from = datetime.datetime(2019,12,3,8,20,0),\
                                 time_to = datetime.datetime(2019,12,3,8,50,0),\
                                 show_visualnames=True)

# in case you want to use my data uncomment this line
unhealthy = pd.read_csv("unhealthy.csv",index_col=0, parse_dates=True)

unhealthy.set_index("TIMESTAMP",drop=True,inplace = True)

# and of course. we will apply the same transformations
unhealthy = unhealthy[['VARIABLE_NAME','VALUE']]
unhealthy['VARIABLE_NAME'] = unhealthy['VARIABLE_NAME'].apply(lambda x: x.replace('RobotSimForPA/Global/',''))
unhealthy = unhealthy.reset_index().pivot(index='TIMESTAMP', columns='VARIABLE_NAME',values='VALUE')
unhealthy_sliced = sliceData(unhealthy,lag)

# Let's plot the first 10 slices
fig,ax = plt.subplots()
for i in range(0,10):
    sns.lineplot(x=unhealthy_sliced.loc[(i,),].index,y=unhealthy_sliced.loc[(i,),'R1_WeldingCurrent'],ax=ax) 
fig.suptitle('unhealthy data - sliced')

# let's compare our mean slices, to see if there is a difference
unhealthy_mean_slice = unhealthy_sliced.mean(level=1)
fig,ax = plt.subplots()
plt.plot(healthy_mean_slice.R1_WeldingCurrent,label='healthy')
plt.plot(unhealthy_mean_slice.R1_WeldingCurrent,label='unhealthy')
fig.legend()
fig.suptitle('mean slices - comparison')


# in this case the difference between healthy and unhealthy data is clearly visible... nevertheless we'll go on

unhealthy_ml = unhealthy_sliced.R1_WeldingCurrent.unstack()

# now we will merge our dataframes and add a label "healthy"/"unhealthy" label
ml_data = healthy_ml.assign(label='healthy')
ml_data = ml_data.append(unhealthy_ml.assign(label='unhealthy'))

# sklearn comes with a handy function to split our data into train and test data
X_train, X_test, y_train, y_test = train_test_split(ml_data.drop(['label'],axis=1), ml_data.label, test_size=0.3, random_state=1)

# create decision tree classifer object
clf = DecisionTreeClassifier()

# train decision tree classifer
clf = clf.fit(X_train,y_train)

# predict the response for test dataset
y_pred = pd.Series(clf.predict(X_test))

# model accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import tree
plt.subplots()
tree.plot_tree(clf)

# as you can see our decision tree is quite simple with only one deciding node.
# In my case this is x[19] <= 208.941. So whenever R1_WeldingCurrent at 19 seconds
# into the cycle is lesser than 208, the cycle is considered unhealthy.
# Our decision tree came to the same result as we did already before with our
# calculation of mean cycles.


# ...and whats your accuracy?! In my case it was 1.0 = 100 % 
# Since our example data is very simple our result is good... with real life
# data this is almost impossible to achieve...



# ------------------------------------------------------------
# ------------------ Part 6 - Use the ML model  --------------
# ------------------------------------------------------------

# to practise our skills at transforming data, we'll load another dataset from
# zenon. This time just load a bigger time frame with randomly mixed healthy and
# unhealthy data.
# We will load and transform that data just like before. Additionally we will
# load data from the ANALYTICS_SUPPORT archive to create a label for our data.
# So we know, for each single welding cycle wether it was healthy or unhealthy.
# Afterwards we will see if our trained ML model will get the same results.


# Get base data and transform it

test_data = zan.read_Online_Archive(project_reference = "PREDICTIVE_MAINTENANCE_DEMO_820",\
                                 archive_reference = "PA",\
                                 variable_references = [\
                                 "RobotSimForPA/Global/R1_WeldingResistance",\
                                 "RobotSimForPA/Global/R1_WeldingCurrent",\
                                 "RobotSimForPA/Global/R1_Status",\
                                 "RobotSimForPA/Global/R1_WeldingVoltage"],\
                                 time_from = datetime.datetime(2019,12,3,9,40,0),\
                                 time_to = datetime.datetime(2019,12,3,10,40,0),\
                                 show_visualnames=True)

# in case you want to use my data uncomment this line
test_data = pd.read_csv("test_data.csv",index_col=0, parse_dates=True)

test_data.set_index("TIMESTAMP",drop=True,inplace=True)

test_data = test_data[['VARIABLE_NAME','VALUE']]
test_data['VARIABLE_NAME'] = test_data['VARIABLE_NAME'].apply(lambda x: x.replace('RobotSimForPA/Global/',''))
test_data = test_data.reset_index().pivot(index='TIMESTAMP', columns='VARIABLE_NAME',values='VALUE')
test_data = test_data[['R1_WeldingCurrent','R1_Status']]

# now load our labes from ANALYTICS_SUPPORT
# In my case, as I used R1 Error 3, I will load R1_Error3_Toggle as an label
# if it is 1 our data is unhealthy

label_data = zan.read_Online_Archive(project_reference = "PREDICTIVE_MAINTENANCE_DEMO_820",\
                                 archive_reference = "AS",\
                                 variable_references = ["RobotSimForPA/ErrorAdditionR1/R1_Error3_Toggle"],\
                                 time_from = datetime.datetime(2019,12,3,9,40,0),\
                                 time_to = datetime.datetime(2019,12,3,10,40,0),\
                                 show_visualnames=True)

label_data.to_csv("label_data.csv")
label_data = pd.read_csv("label_data.csv",index_col=0, parse_dates=True)

label_data.set_index("TIMESTAMP",drop=True,inplace=True)

label_data = label_data[['VARIABLE_NAME','VALUE']]
label_data['VARIABLE_NAME'] = label_data['VARIABLE_NAME'].apply(lambda x: x.replace('RobotSimForPA/ErrorAdditionR1/',''))
label_data = label_data.reset_index().pivot(index='TIMESTAMP', columns='VARIABLE_NAME',values='VALUE')

# now lets merge those to dataframes... easy since indexes are identical
labeled_data= pd.merge(test_data,label_data,left_index=True,right_index=True)

# slice our data
labeled_data_sliced = sliceData(labeled_data,lag)

# get one dataframe with only R1_WeldingCurrent. This is the input to our ml model
labeled_data_ml = labeled_data_sliced.R1_WeldingCurrent.unstack()

# format labels in the same way as before
labels = labeled_data_sliced.mean(level=0).R1_Error3_Toggle.apply(lambda x: 'healthy' if x < 0.5 else 'unhealthy')

# predict...
test_data_prediction = pd.Series(clf.predict(labeled_data_ml))

# and be amazed by the results...
print("Test data accuracy:",metrics.accuracy_score(labels, test_data_prediction))


# again 99.something % for me... 

# This was a very basic exmaple of getting data from zenon and messing around
# with it... of course there is MUCH more to "data science" than this. But I
# hope you got a first impression.

# Thanks for reading!







