import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
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
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

public class TFIDF extends Configured implements Tool {

  private static final Logger LOG = Logger.getLogger(TFIDF.class);

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new TFIDF(), args);
    System.exit(res);
  }

  public int run(String[] args) throws Exception {
    /*Job tf_job = Job.getInstance(getConf(), "termfrequency");
    for (int i = 0; i < args.length; i += 1) {
      if ("-skip".equals(args[i])) {
        tf_job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
        i += 1;
        tf_job.addCacheFile(new Path(args[i]).toUri());
        // this demonstrates logging
        LOG.info("Added file to the distributed cache: " + args[i]);
      }
    }
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
*/
    // TF-IDF
    Job tfidf_job = Job.getInstance(getConf(), "tfidf");
    tfidf_job.setJarByClass(this.getClass());
    // Use TextInputFormat, the default unless job.setInputFormatClass is used
    FileInputFormat.addInputPath(tfidf_job, new Path(args[0]));
    FileOutputFormat.setOutputPath(tfidf_job, new Path(args[1]));
    tfidf_job.setMapperClass(TFIDFMap.class);
    tfidf_job.setReducerClass(TFIDFReduce.class);
    tfidf_job.setOutputKeyClass(Text.class);
    tfidf_job.setOutputValueClass(NullWritable.class);
    int code = tfidf_job.waitForCompletion(true) ? 0 : 1;

    return code;
  }

  public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private boolean caseSensitive = false;
    private long numRecords = 0;
    private String input;
    private Set<String> patternsToSkip = new HashSet<String>();
    private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

    protected void setup(Mapper.Context context)
        throws IOException,
        InterruptedException {
      if (context.getInputSplit() instanceof FileSplit) {
        this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
      } else {
        this.input = context.getInputSplit().toString();
      }
      Configuration config = context.getConfiguration();
      this.caseSensitive = config.getBoolean("wordcount.case.sensitive", false);
      if (config.getBoolean("wordcount.skip.patterns", false)) {
        URI[] localPaths = context.getCacheFiles();
        parseSkipFile(localPaths[0]);
      }
    }

    private void parseSkipFile(URI patternsURI) {
      LOG.info("Added file to the distributed cache: " + patternsURI);
      try {
        BufferedReader fis = new BufferedReader(new FileReader(new File(patternsURI.getPath()).getName()));
        String pattern;
        while ((pattern = fis.readLine()) != null) {
          patternsToSkip.add(pattern);
        }
      } catch (IOException ioe) {
        System.err.println("Caught exception while parsing the cached file '"
            + patternsURI + "' : " + StringUtils.stringifyException(ioe));
      }
    }

    public void map(LongWritable offset, Text lineText, Context context)
        throws IOException, InterruptedException {
      FileSplit fileSplit = (FileSplit)context.getInputSplit();
      String filename = fileSplit.getPath().getName();
      String line = lineText.toString();
      if (!caseSensitive) {
        line = line.toLowerCase();
      }
      Text currentWord = new Text();
      for (String word : WORD_BOUNDARY.split(line)) {
        if (word.isEmpty() || patternsToSkip.contains(word)) {
            continue;
        }
            currentWord = new Text(word + "#####" + filename);
            context.write(currentWord, one);
        }             
    }
  }

  public static class TFIDFMap extends Mapper<LongWritable, Text, Text, NullWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private boolean caseSensitive = false;
    private long numRecords = 0;
    private String input;
    private Set<String> patternsToSkip = new HashSet<String>();
    private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

    protected void setup(Mapper.Context context)
        throws IOException,
        InterruptedException {
      if (context.getInputSplit() instanceof FileSplit) {
        this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
      } else {
        this.input = context.getInputSplit().toString();
      }
      Configuration config = context.getConfiguration();
      this.caseSensitive = config.getBoolean("wordcount.case.sensitive", false);
      /*if (config.getBoolean("wordcount.skip.patterns", false)) {
        URI[] localPaths = context.getCacheFiles();
        parseSkipFile(localPaths[0]);
      }*/
    }

    private void parseSkipFile(URI patternsURI) {
      LOG.info("Added file to the distributed cache: " + patternsURI);
      try {
        BufferedReader fis = new BufferedReader(new FileReader(new File(patternsURI.getPath()).getName()));
        String pattern;
        while ((pattern = fis.readLine()) != null) {
          patternsToSkip.add(pattern);
        }
      } catch (IOException ioe) {
        System.err.println("Caught exception while parsing the cached file '"
            + patternsURI + "' : " + StringUtils.stringifyException(ioe));
      }
    }

    public static String angularOutput(String key, String fileName, int count) {
      StringBuilder sb = new StringBuilder();
      sb.append("<\"").append(key)
              .append("\",\"").append(fileName)
              .append("=").append(count).append("\">");
      return sb.toString();
    }
    public void map(LongWritable offset, Text lineText, Context context)
        throws IOException, InterruptedException {
      FileSplit fileSplit = (FileSplit)context.getInputSplit();
      String filename = fileSplit.getPath().getName();
      String line = lineText.toString();
      if (!caseSensitive) {
        line = line.toLowerCase();
      }
      System.out.println("Line: " + line);
      Text currentWord = new Text();
        if (line.isEmpty()) {
            return;
        }
        currentWord = new Text(angularOutput(line, filename, 1));
        context.write(currentWord, NullWritable.get());

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
      context.write(word, new DoubleWritable(1.00+Math.log(sum)));
    }

  }

  public static class TFIDFReduce extends Reducer<Text, NullWritable, Text, NullWritable> {
    @Override
    public void reduce(Text word, Iterable<NullWritable> counts, Context context)
        throws IOException, InterruptedException {
      context.write(word, NullWritable.get());
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
}

