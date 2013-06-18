in-memory-cooccurrence
======================

Analyze occurrence data for significant cooccurrence using Mahout sparse matrices.  This is a single threaded 
illustration of some of the algorithms used in Mahout.  The point of this is to illustrate the algorithms in
a simple form and to allow small problems (up to say a billion occurrences) to be computed without having to
go to the trouble of setting up a Hadoop cluster.

To compile and package this code into a single jar, use this command

    mvn package

To run the analyzer on a file *f*,

    java -jar target/in-memory-cooccurrence-0.1-SNAPSHOT-jar-with-dependencies.jar f
    
To work on larger files, add a heap size adjustment such as -Xmx40G.


## Testing

Before running the scale tests, you will need to generate the data for the tests.  This will take some time for
larger data sets.

To generate data with about 10^n1 and 10^n2 data points, use this command:

    java -cp target/in-memory-cooccurrence-0.1-SNAPSHOT-jar-with-dependencies.jar com.tdunning.cooc.GenerateData n1 n2

You can include as many scale values as you like.