/***
 * Md Maruf Hossain
 * mhossa10@uncc.edu
 **/

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.math.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.regex.Pattern;

public class Search extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(Search.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Search(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        /// Take the input from user for search
        Scanner input = new Scanner(System.in);
        System.out.println("Enter Search Strings:");
        /// Add the search string in the configuaration to process at Mapper class
        conf.set("SEARCH_STRINGS", input.nextLine());
        /// creating the job
        Job job = Job.getInstance(conf, "search");
        job.setJarByClass(this.getClass());
        /// Get the input file from the first parameter and add to the input path
        FileInputFormat.addInputPath(job, new Path(args[0]));
        /// Get the output file location from the second parameter and add to the output path
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        /// Add Mapper Class
        job.setMapperClass(Map.class);
        /// Add Combiner Class
        job.setCombinerClass(Combine.class);
        /// Add Reduce Class
        job.setReducerClass(Reduce.class);
        /// Set intermediate output key as Text
        job.setOutputKeyClass(Text.class);
        /// Set intermediate output value as Double format
        job.setOutputValueClass(DoubleWritable.class);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    /// Map Class
    public static class Map extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        private final static IntWritable one = new IntWritable(1);
        private boolean caseSensitive = false;
        private String input;
        /// Pattern to split the input line
        private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

        /// Setup the input and case sensitivity
        protected void setup(Mapper.Context context)
                throws IOException,
                InterruptedException {
            if (context.getInputSplit() instanceof FileSplit) {
                this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
            } else {
                this.input = context.getInputSplit().toString();
            }
            /// Make case sensitive false for this project
            this.caseSensitive = false;
        }

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            /// get the input line as string and trim it.
            String line = lineText.toString();
            line = line.trim();
            /// Retrieve the search strings
            String searchItems = context.getConfiguration().get("SEARCH_STRINGS");
            if (!caseSensitive) {
                /// make it lower case
                line = line.toLowerCase();
                searchItems = searchItems.toLowerCase();
            }
            if (line.isEmpty()) {
                return;
            }
            /// Split the line by "#####"
            String[] tokens = line.split("#####");

            if (!tokens[0].isEmpty() && tokens.length >= 2) {
                boolean matchFound = false;
                for (String searchWord : WORD_BOUNDARY.split(searchItems)) {
                    /// Check search string contains the whole word
                    if (searchWord.equalsIgnoreCase(tokens[0])) {
                        matchFound = true;
                        break;
                    }
                }
                /// Check if match found
                if (matchFound) {
                    /// split file name and value of the TF-IDF
                    String[] values = tokens[1].split("\\s");
                    if (values.length >= 2) {
                        /// Output file name and TF-IDF score
                        context.write(new Text(values[0]), new DoubleWritable(Double.valueOf(values[1])));
                    }
                }
            }
        }
    }

    /// Reduce class
    public static class Reduce extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double TFIDF_Score = 0;
            for (DoubleWritable val : values) {
                /// Sum all the TF-IDF score
                TFIDF_Score += val.get();
            }
            context.write(new Text(word), new DoubleWritable(TFIDF_Score));
        }
    }

    /// Combiner class
    public static class Combine extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double TFIDF_Score = 0;
            for (DoubleWritable val : values) {
                /// Sum all the TF-IDF score
                TFIDF_Score += val.get();
            }
            context.write(word, new DoubleWritable(TFIDF_Score));
        }
    }
}

