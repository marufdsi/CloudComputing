input=/user/cloudera/wordcount/input
docoutput=/user/cloudera/wordcount/docoutput
tfoutput=/user/cloudera/wordcount/tfoutput
tfidfoutput=/user/cloudera/wordcount/tfidfoutput
searchoutput=/user/cloudera/wordcount/searchoutput

run: clean-dw build-docwordcount docwordcount.jar jar
	hadoop fs -rm -f -r  $(docoutput)
	hadoop jar docwordcount.jar DocWordCount $(input) $(docoutput)

run-tf: clean-tf build-termfrequency termfrequency.jar tfjar
	hadoop fs -rm -f -r  /user/cloudera/wordcount/tfoutput
	hadoop jar termfrequency.jar TermFrequency /user/cloudera/wordcount/input /user/cloudera/wordcount/tfoutput

run-tfidf: clean-tfidf build-tfidf tfidf.jar tfidfjar
	hadoop fs -rm -f -r  /user/cloudera/wordcount/tf_output
	hadoop fs -rm -f -r  /user/cloudera/wordcount/tfidfoutput
	hadoop jar tfidf.jar TFIDF /user/cloudera/wordcount/input /user/cloudera/wordcount/tf_output /user/cloudera/wordcount/tfidfoutput 8

run-search: clean-search build-search search.jar searchjar
	hadoop fs -rm -f -r  /user/cloudera/wordcount/searchoutput
	hadoop jar search.jar Search /user/cloudera/wordcount/tfidfoutput /user/cloudera/wordcount/searchoutput

run_caseSensitive: jar
	hadoop fs -rm -f -r  /user/cloudera/wordcount/docoutput
	hadoop jar wordcount.jar org.myorg.WordCount -Dwordcount.case.sensitive=true /user/cloudera/wordcount/input /user/cloudera/wordcount/docoutput

run_stopwords: jar stopwords
	hadoop fs -rm -f -r  /user/cloudera/wordcount/docoutput
	hadoop jar docwordcount.jar DocWordCount /user/cloudera/wordcount/input /user/cloudera/wordcount/docoutput -skip /user/cloudera/wordcount/stop_words.text

compile: build/org/myorg/DocWordCount.class

jar: docwordcount.jar

tfjar: termfrequency.jar

tfidfjar: tfidf.jar

searchjar: search.jar

docwordcount.jar: build/docwordcount/DocWordCount.class
	jar -cvf docwordcount.jar -C build/docwordcount .

build-docwordcount: DocWordCount.java
	mkdir -p build/docwordcount
	javac -cp /usr/lib/hadoop/*:/usr/lib/hadoop-mapreduce/* DocWordCount.java -d build/docwordcount -Xlint

termfrequency.jar: build/termfrequency/TermFrequency.class
	jar -cvf termfrequency.jar -C build/termfrequency .

build-termfrequency: TermFrequency.java
	mkdir -p build/termfrequency
	javac -cp /usr/lib/hadoop/*:/usr/lib/hadoop-mapreduce/* TermFrequency.java -d build/termfrequency -Xlint

tfidf.jar: build/tfidf/TFIDF.class
	jar -cvf tfidf.jar -C build/tfidf .

build-tfidf: TFIDF.java
	mkdir -p build/tfidf
	javac -cp /usr/lib/hadoop/*:/usr/lib/hadoop-mapreduce/* TFIDF.java -d build/tfidf -Xlint

search.jar: build/search/Search.class
	jar -cvf search.jar -C build/search .

build-search: Search.java
	mkdir -p build/search
	javac -cp /usr/lib/hadoop/*:/usr/lib/hadoop-mapreduce/* Search.java -d build/search -Xlint

clean:
	rm -rf build docwordcount.jar
	rm -rf build termfrequency.jar

clean-dw:
	rm -rf build/docwordcount docwordcount.jar

clean-tf:
	rm -rf build/termfrequency termfrequency.jar

clean-tfidf:
	rm -rf build/tfidf tfidf.jar

clean-search:
	rm -rf build/search search.jar

data:
	hadoop fs -rm -f -r /user/cloudera/wordcount/input
	hadoop fs -mkdir /user/cloudera/wordcount
	hadoop fs -mkdir /user/cloudera/wordcount/input
	echo "Hadoop is an elephant" > file0
	echo "Hadoop is as yellow as can be" > file1
	echo "Oh what a yellow fellow is Hadoop" > file2
	hadoop fs -put file* /user/cloudera/wordcount/input
	rm file*

poetry:
	hadoop fs -rm -f -r /user/cloudera/wordcount/input
	hadoop fs -mkdir /user/cloudera/wordcount/input
	echo -e "Hadoop is the Elephant King! \\nA yellow and elegant thing.\\nHe never forgets\\nUseful data, or lets\\nAn extraneous element cling! "> HadoopPoem0.txt
	echo -e "A wonderful king is Hadoop.\\nThe elephant plays well with Sqoop.\\nBut what helps him to thrive\\nAre Impala, and Hive,\\nAnd HDFS in the group." > HadoopPoem1.txt
	echo -e "Hadoop is an elegant fellow.\\nAn elephant gentle and mellow.\\nHe never gets mad,\\nOr does anything bad,\\nBecause, at his core, he is yellow." > HadoopPoem2.txt
	hadoop fs -put HadoopP* /user/cloudera/wordcount/input
	rm HadoopPoem*
gettfoutput:
	rm -r /home/cloudera/Assignment2/tf_output.out
	hadoop fs -get  /user/cloudera/wordcount/tf_output/part-r-00000  /home/cloudera/Assignment2/tf_output.out

gettfidfoutput:
	rm -r /home/cloudera/Assignment2/tfidfoutput.out
	hadoop fs -get  /user/cloudera/wordcount/tfidfoutput/part-r-00000  /home/cloudera/Assignment2/tfidfoutput.out

showResult:
	hadoop fs -cat /user/cloudera/wordcount/output/*
	
stopwords:
	hadoop fs -rm -f /user/cloudera/wordcount/stop_words.text
	echo -e "a\\nan\\nand\\nbut\\nis\\nor\\nthe\\nto\\n.\\n," >stop_words.text
	hadoop fs -put stop_words.text /user/cloudera/wordcount/

