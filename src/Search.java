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
        Scanner input = new Scanner(System.in);
        System.out.println("Enter Search Strings:");
        conf.set("SEARCH_STRINGS", input.nextLine());

        Job job = Job.getInstance(conf, "search");
        job.setJarByClass(this.getClass());
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        job.setMapperClass(Map.class);
        job.setCombinerClass(Combine.class);
        job.setReducerClass(Reduce.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class Map extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        private final static IntWritable one = new IntWritable(1);
        private boolean caseSensitive = false;
        private String input;
        private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

        protected void setup(Mapper.Context context)
                throws IOException,
                InterruptedException {
            if (context.getInputSplit() instanceof FileSplit) {
                this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
            } else {
                this.input = context.getInputSplit().toString();
            }
            this.caseSensitive = false;
        }

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String filename = fileSplit.getPath().getName();
            String line = lineText.toString();
            line = line.trim();
            String searchItems = context.getConfiguration().get("SEARCH_STRINGS");
            if (!caseSensitive) {
                line = line.toLowerCase();
                searchItems = searchItems.toLowerCase();
            }
            if (line.isEmpty()) {
                return;
            }
            String[] tokens = line.split("#####");

            if (!tokens[0].isEmpty() && tokens.length>=2) {
                boolean matchFound = false;
                for(String searchWord: WORD_BOUNDARY.split(searchItems)){
                    if (searchWord.equalsIgnoreCase(tokens[0])){
                        matchFound = true;
                        break;
                    }
                }
                if (matchFound) {
                    String[] values = tokens[1].split("\\s");
                    if (values.length >= 2)
                        context.write(new Text(values[0]), new DoubleWritable(Double.valueOf(values[1])));
                }
            }
        }
    }

    public static class Reduce extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double TFIDF_Score = 0;
            for (DoubleWritable val : values) {
                TFIDF_Score += val.get();
            }
            context.write(new Text(word), new DoubleWritable(TFIDF_Score));
        }
    }

    public static class Combine extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            double TFIDF_Score = 0;
            for (DoubleWritable val : values) {
                TFIDF_Score += val.get();
            }
            context.write(word, new DoubleWritable(TFIDF_Score));
        }
    }
}

