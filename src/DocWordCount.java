/***
 * Md Maruf Hossain
 * mhossa10@uncc.edu
 **/

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URI;
import java.util.HashSet;
import java.util.Set;
import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.StringUtils;

import org.apache.log4j.Logger;

public class DocWordCount extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(DocWordCount.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new DocWordCount(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        /// creating the job
        Job job = Job.getInstance(getConf(), "docwordcount");
        job.setJarByClass(this.getClass());
        /// Get the input file from the first parameter and add to the input path
        FileInputFormat.addInputPath(job, new Path(args[0]));
        /// Get the output file location from the second parameter and add to the output path
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        /// Add Mapper Class
        job.setMapperClass(Map.class);
        /// Add Combiner Class, in this case use Reduce class because both of them are doing same task
        job.setCombinerClass(Reduce.class);
        /// Add Reduce Class
        job.setReducerClass(Reduce.class);
        /// Set output key as Text
        job.setOutputKeyClass(Text.class);
        /// Set intermediate output value as Integer format
        job.setOutputValueClass(IntWritable.class);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    /// Map Class
    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
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
            Configuration config = context.getConfiguration();
            /// Make case sensitive false for this project
            this.caseSensitive = false;
        }

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            /// Get the file name, because we need it for the output
            String filename = fileSplit.getPath().getName();
            /// get the input line as string and trim it.
            String line = lineText.toString();
            line = line.trim();
            if (!caseSensitive) {
                /// make it lower case
                line = line.toLowerCase();
            }
            Text currentWord = new Text();
            for (String word : WORD_BOUNDARY.split(line)) {
                word = word.trim();
                /// Check the input line is empty or does not conatain any alphabet
                if (word.isEmpty() || !word.matches(".*[a-zA-Z]+.*")) {
                    continue;
                }
                /// Create desired outout format
                currentWord = new Text(word + "#####" + filename);
                context.write(currentWord, one);
            }
        }
    }

    /// Reduce class as well as Combiner class
    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text word, Iterable<IntWritable> counts, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable count : counts) {
                /// Sum all the count
                sum += count.get();
            }
            context.write(word, new IntWritable(sum));
        }
    }
}

