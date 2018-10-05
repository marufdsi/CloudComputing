input=/user/cloudera/wordcount/input
docoutput=/user/cloudera/wordcount/docoutput
tfoutput=/user/cloudera/wordcount/tfoutput
tfidfoutput=/user/cloudera/wordcount/tfidfoutput
searchoutput=/user/cloudera/wordcount/searchoutput
NUMBER_OF_DOCUMENT=8

run-doc: clean-dw build-docwordcount doc-jar
	hadoop fs -rm -f -r  $(docoutput)
	hadoop jar docwordcount.jar DocWordCount $(input) $(docoutput)

run-tf: clean-tf build-termfrequency tf-jar
	hadoop fs -rm -f -r  $(tfoutput)
	hadoop jar termfrequency.jar TermFrequency $(input) $(tfoutput)

run-tfidf: clean-tfidf build-tfidf tfidf-jar
	hadoop fs -rm -f -r  $(tfoutput)
	hadoop fs -rm -f -r  $(tfidfoutput)
	hadoop jar tfidf.jar TFIDF $(input) $(tfoutput) $(tfidfoutput) $(NUMBER_OF_DOCUMENT)

run-search: clean-search build-search search-jar
	hadoop fs -rm -f -r  $(searchoutput)
	hadoop jar search.jar Search $(tfidfoutput) $(searchoutput)


doc-jar: docwordcount.jar

tf-jar: termfrequency.jar

tfidf-jar: tfidf.jar

search-jar: search.jar

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
	rm -rf build docwordcount.jar termfrequency.jar tfidf.jar search.jar
	hadoop fs -rm -f -r  $(docoutput)
	hadoop fs -rm -f -r  $(tfoutput)
	hadoop fs -rm -f -r  $(tfidfoutput)
	hadoop fs -rm -f -r  $(searchoutput)

clean-dw:
	rm -rf build/docwordcount docwordcount.jar

clean-tf:
	rm -rf build/termfrequency termfrequency.jar

clean-tfidf:
	rm -rf build/tfidf tfidf.jar

clean-search:
	rm -rf build/search search.jar



