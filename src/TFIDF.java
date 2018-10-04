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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

public class TFIDF extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(TFIDF.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new TFIDF(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        conf.set("NUMBER_OF_FILE", args[3]);
        Job tf_job = Job.getInstance(conf, "termfrequency");
        tf_job.setJarByClass(this.getClass());
        // Use TextInputFormat, the default unless job.setInputFormatClass is used
        FileInputFormat.addInputPath(tf_job, new Path(args[0]));
        FileOutputFormat.setOutputPath(tf_job, new Path(args[1]));
        tf_job.setMapperClass(Map.class);
        tf_job.setCombinerClass(Combine.class);
        tf_job.setReducerClass(Reduce.class);
        tf_job.setOutputKeyClass(Text.class);
        tf_job.setOutputValueClass(IntWritable.class);
        int code = tf_job.waitForCompletion(true) ? 0 : 1;

        // TF-IDF
        Job tfidf_job = Job.getInstance(conf, "tfidf");
        tfidf_job.setJarByClass(this.getClass());
        FileInputFormat.addInputPath(tfidf_job, new Path(args[1]));
        FileOutputFormat.setOutputPath(tfidf_job, new Path(args[2]));
        tfidf_job.setMapperClass(TFIDFMap.class);
//        tfidf_job.setCombinerClass(TFIDFCombine.class);
        tfidf_job.setReducerClass(TFIDFReduce.class);
        tfidf_job.setOutputKeyClass(Text.class);
        tfidf_job.setOutputValueClass(Text.class);
        code = tfidf_job.waitForCompletion(true) ? 0 : 1;

        return code;
    }

    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private boolean caseSensitive = false;
        private Set<String> patternsToSkip = new HashSet<String>();
        private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");
        private String input;
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
            if (!caseSensitive) {
                line = line.toLowerCase();
            }
            Text currentWord = new Text();
            for (String word : WORD_BOUNDARY.split(line)) {
                word = word.trim();
                if (word.isEmpty() || patternsToSkip.contains(word) || !word.matches(".*[a-zA-Z]+.*")) {
                    continue;
                }
                currentWord = new Text(word + "#####" + filename);
                context.write(currentWord, one);
            }
        }
    }

    public static class TFIDFMap extends Mapper<LongWritable, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);
        private boolean caseSensitive = false;
        private static final Pattern WORD_BOUNDARY = Pattern.compile("#####");
        private String input;
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
            String line = lineText.toString();
            line = line.trim();
            if (!caseSensitive) {
                line = line.toLowerCase();
            }
            if (line.isEmpty()) {
                return;
            }

            String[] tokens = WORD_BOUNDARY.split(line);
            if (tokens.length>=2) {
               // context.write(new Text(tokens[0]), new Text(tokens[1]));
                String[] values = tokens[1].trim().split("\\s+");
                if(values.length>=2)
                    context.write(new Text(tokens[0]), new Text(values[0] + "=" + values[1]));
            }
        }
    }

    public static class Reduce extends Reducer<Text, IntWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<IntWritable> counts, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable count : counts) {
                sum += count.get();
            }
            double TF = 1.00 + Math.log10(sum);
            if (TF<0){
                TF = 0.0;
            }
            context.write(word, new DoubleWritable(TF));
        }
    }

    public static class TFIDFReduce extends Reducer<Text, Text, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            long totalNumberOfDoc = Long.parseLong(context.getConfiguration().get("NUMBER_OF_FILE"));
            long numberOfDocContain = 0;
            List<String> cache = new ArrayList<String>();
            for (Text val : values) {
                numberOfDocContain++;
                cache.add(val.toString());
            }
            if (numberOfDocContain == 0)
                return;
            double IDF = Math.log10(1 + (totalNumberOfDoc/numberOfDocContain));
            for (int i=0; i<cache.size(); ++i) {
                String[] tokens = cache.get(i).split("=");
                if (tokens.length>=2)
                    context.write(new Text(word + "#####" + tokens[0]), new DoubleWritable(Double.valueOf(tokens[1])*IDF));
            }
        }
    }

    public static class Combine extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text word, Iterable<IntWritable> counts, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable count : counts) {
                sum += count.get();
            }
            context.write(word, new IntWritable(sum));
        }
    }

  /*public static class TFIDFCombine extends Reducer<Text, Text, Text, Text> {
    @Override
    public void reduce(Text word, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
      for (Text val : values) {
        context.write(word, val);
      }
    }
  }*/
}

