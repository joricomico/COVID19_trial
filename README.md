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

THIS REPOSITORY is OBSOLETE, a NEW REPOSITORY will be introduced very SOON.

RAWBASE will not only provide a framework to analyze subjects' data as time series or through classical categorical machine learning analysis, but it will also introduce a NoSQL database that will facilitate text parsing and the construction of tables.

Some examples from the upcoming repository:

main = Table() #creates a new table

Add(main, line(id=1, var1='abc', var2='test'), line(id=2, var2='3'), line(id=1, rawdate='OCT78')) #add some lines to the table

main #in ipython or Jupyter or interactive python, repr(main) in other cases, lists all variables introduced in the table, showing some statistics...

id:       3 <int> 100%	1..2 (1.15±0.504)

var1:     1 <str> 100%	abc

var2:     2 <int> 50%	3

rawdate:  1 <Time> 100%	Sun 01/10/1978

#... prevalent or numerical type is shown when data field has different kind of associations, like "var2"
 
date = main.expand(date=('rawdate', TryParse)) #expands the table creating a new field "date" whenever a field "rawdate" is in the line...

date      #... TryParse translate numbers and dates to the corresponding types.

id:       3 <int> 100%	1..2 (1.15±0.504)

var1:     1 <str> 100%	abc

var2:     2 <int> 50%	3

rawdate:  1 <Time> 100%	Sun 01/10/1978

date:     1 <Time> 100%	Sun 01/10/1978

#... even though "rawdate" and "date" seem to hold the same data, "rawdate" is a text field, while "date" is associated to the Time class which unifies several python classes managing time in python...

A = Time.parse('24/10/78', "%d/%m/%y")

B = Time.today()

C = Time(ms=10)

print(A>B, A<B, A==B, A!=B, A>=B, A<=B)

False True False True False True

C         #is a delta

delta: 0 days, 0h 0m 0s 10ms 0us, 0.01 total secs

A.is_date #and B are very different, all can be compared when translated to deltas

True


#Apart from date and time, the framework will offer several commands to parse and compare text...

compareEntries("this", "that") #.5

compareEntries("this", "this is strange") #.541666666

bidirectionalCompare("this", "this is strange") #.343055555

text = "jump this, take value=1 here, but not value=2 here!"

greedy = rule('take', TX(""), n=GREEDY)

gbound = rule('take here', n=GREEDY)

Match(text, greedy) #0:	<core.text.Entry> take value=1 here, but not value=2 here!

Match(text, gbound) #0:	<core.text.Entry> take value=1 here

  
#going back to tables, as they can be expanded, they can also be reduced...

main.to(id="ID", var1='') == main.to(DELETE, id="ID", var2='', rawdate='') #... by either selecting (and/or translating) the fields to keep, or by deleting them

True

date.select(date=lambda x: x>Time(M=9, Y=1978)) #moreover, tables can be filtered by creating rules...

id:       1 <int> 100%	1

rawdate:  1 <Time> 100%	Sun 01/10/1978

date:     1 <Time> 100%	Sun 01/10/1978
  
#AND MANY OTHER commands and tools, also to upload text data and tables...

  
