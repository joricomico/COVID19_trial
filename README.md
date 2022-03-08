# COVID19_trial
bundle to run clinical trials with COP applied on COVID19

covid19 examples

M, R = test('vars.txt', 'file.csv', FOR.MechVent, err0=.15, retry=2) #tests a new model from data in csv file, using variables in vars.txt
  #the model is based on 2 repetitions (retry), set to 0 to just use general sets for training;
  #use class FOR for setting the prediction level, see class FOR in code and refer to the main paper for classes;
  #err0 is the balancing threshold start, automatic threshold is on.
oR = organize_result(M,R) #organizes and shows results
train_tested(M) #trains the selected model M with all data available.

In next versions:
- a function to predict a series of new patients will be introduced;
- selective training depending on time limits;
- an example of graphical UI to insert data.

THIS REPOSITORY will soon become OBSOLETE, a new REPOSITORY will be introduced very SOON.

RAWBASE will not only provide a framework to analyze subjects' data as time series or through classical categorical machine learning analysis, but it will also introduce a NoSQL database that will facilitate text parsing and the construction of tables.

Some examples from the upcoming repository:

