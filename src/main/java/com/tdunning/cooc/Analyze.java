package com.tdunning.cooc;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.*;
import com.google.common.io.CharStreams;
import com.google.common.io.Files;
import com.google.common.io.InputSupplier;
import com.google.common.io.LineProcessor;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.LogLikelihood;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * Analyzes data in a sparse matrix for interesting cooccurrence.
 * <p/>
 * The input is assumed to contain two tab separated fields which are string names
 * of the row and column of the cooccurrence matrix, respectively.  The output
 * of this method is a reduced cooccurrence matrix suitable for indexing.
 * <p/>
 * The process of reduction involves:
 * <p/>
 * <ul>
 * <li>downsampling the data to limit row and column cardinality</li>
 * <li>using LLR to find anomalous cooccurrence</li>
 * </ul>
 * <p/>
 * This is done by making several passes over an input file.  The first pass is used
 * to create dictionaries of the symbols involved.  The second pass populates the
 * cooccurrence matrix while down-sampling the data.
 */
public class Analyze {
    public static final Splitter onTab = Splitter.on("\t");
    private Logger log = LoggerFactory.getLogger(Analyze.class);

    private Dictionary rowDict;
    private Dictionary colDict;
    private Matrix filteredMatrix;
    private Multimap<Integer, Integer> referenceSentences;

    /**
     * This is for illustration purposes only at this time.
     *
     * @param args A file to process
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        Options options = parseOptions(args);
        Preconditions.checkArgument(options.files.size() == 1, "Should only have one file argument");

        Analyze analyzer = new Analyze(Files.newReaderSupplier(new File(options.files.get(0)), Charsets.UTF_8),
                options.maxRowCount, options.maxColumnCount, options.maxRelated);


        Matrix m = analyzer.getFilteredCooccurrence();
        Dictionary colDict = analyzer.getColDict();
        Dictionary rowDict = analyzer.getRowDict();
        Multimap<Integer, Integer> rSentences = analyzer.getReferenceSentences();

        int d = m.columnSize();
        for (MatrixSlice row : m) {
            for (Vector.Element element : row.vector().nonZeroes()) {
                int z = row.index() * d + element.index();
                Collection<Integer> concept = rSentences.get(z);
                Iterator<Integer> it = concept.iterator();
                System.out.printf("%s\t%s\t%s", colDict.values().get(row.index()), colDict.values().get(element.index()), concept.size());
                while (it.hasNext()) {
                    System.out.printf("\t%s", rowDict.values().get(it.next()));
                }
                System.out.printf("\n");
            }
        }
    }

    /**
     * Package local constructor for testing only.
     */
    Analyze() {
        rowDict = new Dictionary();
        colDict = new Dictionary();
    }

    /**
     * Analyze an input to find significant cooccurrences.  The input should consist of tab-delimited
     * row and column designators.  The row is the unit of text for textual cooccurrence or the user
     * for recommendations.  The output will be a matrix of significant cooccurrences.
     *
     * @param input          A tab-delimited input file with row and column descriptors
     * @param maxRowCount    How many elements in a single row of the occurrence matrix should be retained?
     * @param maxColumnCount How many columns should be retained?
     * @param maxRelated     How many related items should be retained?
     * @throws IOException
     */
    public Analyze(InputSupplier<InputStreamReader> input, double maxRowCount, double maxColumnCount, int maxRelated) throws IOException {
        this();

        Matrix occurrences = readOccurrenceMatrix(input, maxRowCount, maxColumnCount);
        File sortedData = java.nio.file.Files.createTempFile("raw", ".dat").toFile();

        // Write the occurrence data out in row major order
        LineCounter counter = new LineCounter(log, "Writing sorted data");
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(sortedData))) {
            out.writeInt(occurrences.rowSize());
            out.writeInt(occurrences.columnSize());
            for (MatrixSlice row : occurrences) {
                counter.step();
                out.writeInt(Iterables.size(row.nonZeroes()));
                for (Vector.Element element : row.nonZeroes()) {
                    out.writeInt(element.index());
                }
            }
        }
        log.info("Sorted data is {} MB", String.format("%.1f", sortedData.length() / 1e6));

        // now we square the occurrence matrix to get cooccurrences
        // by reading the data from the file, we don't have to keep both cooccurrence and original data in memory
        Matrix cooccurrence = squareFile(sortedData);

        log.info("Starting sums");
        // to determine anomalous cooccurrence, we need row and column sums
        Vector rowSums = new DenseVector(cooccurrence.rowSize());
        Vector colSums = new DenseVector(cooccurrence.columnSize());
        for (MatrixSlice row : cooccurrence) {
            rowSums.set(row.index(), row.vector().zSum());
            colSums.assign(row, Functions.PLUS);
        }
        log.info("Largest row sum = {}", rowSums.maxValue());
        log.info("Largest column sum = {}", colSums.maxValue());

        // and the total
        double total = rowSums.zSum();

        log.info("Scoring");
        for (MatrixSlice row : cooccurrence) {
            for (Vector.Element element : row.vector().nonZeroes()) {
                long k11 = (long) element.get();
                long k12 = (long) (rowSums.get(row.index()) - k11);
                long k21 = (long) (colSums.get(element.index()) - k11);
                long k22 = (long) (total - k11 - k12 - k21);
                double score = LogLikelihood.rootLogLikelihoodRatio(k11, k12, k21, k22);
                element.set(score);
            }
        }

        log.info("Filtering");
        filteredMatrix = new SparseRowMatrix(cooccurrence.rowSize(), cooccurrence.columnSize(), true);
        for (MatrixSlice row : cooccurrence) {
            List<Vector.Element> elements = Lists.newArrayList(row.vector().nonZeroes());
            Collections.sort(elements, new Ordering<Vector.Element>() {
                public int compare(Vector.Element o1, Vector.Element o2) {
                    return Double.compare(o1.get(), o2.get());
                }
            }.reverse());
            elements = elements.subList(0, Math.min(maxRelated, elements.size()));
            for (Vector.Element element : elements) {
                if (element.get() > 0) {
                    filteredMatrix.set(row.index(), element.index(), 1);
                }
            }
        }
        log.info("Done");

    }

    public Matrix readOccurrenceMatrix(InputSupplier<InputStreamReader> input, final double maxRowCount, final double maxColumnCount) throws IOException {
        final Multiset<Integer> rowCounts = HashMultiset.create();
        final Multiset<Integer> colCounts = HashMultiset.create();

        log.info("Starting first pass");
        // read the data and build dictionaries.  This tells us how large the occurrence matrix must be
        CharStreams.readLines(input, new LineProcessor<Object>() {
            LineCounter counter = new LineCounter(log);

            public boolean processLine(String s) throws IOException {
                counter.step();

                Iterator<String> x = onTab.split(s).iterator();
                String s1 = x.next();
                String s2 = x.next();
                rowCounts.add(rowDict.intern(s1));
                colCounts.add(colDict.intern(s2));
                return true;
            }

            public Object getResult() {
                return null;
            }
        });
        log.info("Average non-zeros per row {}", (double) rowCounts.size() / rowCounts.elementSet().size());


        // now we can read the actual data.  Note that we downsample this data based on our first pass
        log.info("Starting second pass");
        return CharStreams.readLines(input, new LineProcessor<Matrix>() {
            Random random = RandomUtils.getRandom();
            Matrix m = new SparseRowMatrix(rowDict.size(), colDict.size(), true);
            LineCounter counter = new LineCounter(log);
            public double minSampleRate = Double.MAX_VALUE;
            final int[] stats = {0, 0};

            public boolean processLine(String s) throws IOException {
                counter.step();
                Iterator<String> x = onTab.split(s).iterator();
                int row = rowDict.intern(x.next());
                int col = colDict.intern(x.next());

                double rowSampleRate = Math.min(maxRowCount, rowCounts.count(row)) / rowCounts.count(row);
                double columnSampleRate = Math.min(maxColumnCount, colCounts.count(col)) / colCounts.count(col);
                minSampleRate = Math.min(minSampleRate, Math.min(rowSampleRate, columnSampleRate));

                // this down-samples at a rate of Math.min(rowRate, colRate).  The alternative would be
                // to generate two random numbers so we could downsample at the rate of rowRate * colRate.
                // Since one of those should almost always be == 1, this shouldn't much matter.
                if (random.nextDouble() <= Math.min(rowSampleRate, columnSampleRate)) {
                    m.set(row, col, 1);
                    stats[0]++;
                }
                stats[1]++;
                return true;
            }

            public Matrix getResult() {
                log.info("Done with second pass");
                log.info("Retained {} / {} elements", stats[0], stats[1]);
                log.info("Minimum sample factor {}", minSampleRate);
                return m;
            }
        });
    }

    /**
     * Given a matrix A, return A' * A which can be interpreted as a cooccurrence matrix.
     * <p/>
     * Exposed for testing only.
     *
     * @param occurrences The original occurrence data
     * @return The cooccurrence data.
     */
    Matrix square(Matrix occurrences) {
        LineCounter counter = new LineCounter(log);

        log.info("Starting cooccurrence counting");
        Matrix r = new SparseRowMatrix(occurrences.columnSize(), occurrences.columnSize());
        for (MatrixSlice row : occurrences) {
            counter.step();
            for (Vector.Element e1 : row.nonZeroes()) {
                for (Vector.Element e2 : row.nonZeroes()) {
                    if (e1.index() != e2.index()) {
                        r.set(e1.index(), e2.index(), r.get(e1.index(), e2.index()) + e1.get() * e2.get());
                        r.set(e2.index(), e1.index(), r.get(e1.index(), e2.index()));
                    }
                }
            }
        }
        return r;
    }

    /**
     * Given a binary matrix A sorted in a file as raw integers, compute all cooccurrences in memory.
     *
     * @param occurrences The file containing the original occurrence data
     * @return The cooccurrence data.
     */
    Matrix squareFile(File occurrences) throws IOException {
        LineCounter counter = new LineCounter(log, "Cooc");

        log.info("Starting cooccurrence counting");
        referenceSentences = ArrayListMultimap.create();

        try (DataInputStream in = new DataInputStream(new FileInputStream(occurrences))) {
            int rowCount = in.readInt();
            int columnCount = in.readInt();
            Matrix r = new SparseRowMatrix(columnCount, columnCount);
            for (int row = 0; row < rowCount; row++) {
                counter.step();
                int nonZeroCount = in.readInt();
                int[] columns = new int[nonZeroCount];

                for (int j = 0; j < nonZeroCount; j++) {
                    columns[j] = in.readInt();
                }

                for (int n : columns) {
                    for (int m : columns) {
                        if (n != m) {
                            double newValue = r.get(n, m) + 1;
                            r.set(n, m, newValue);

                            //Adriano: Store Reference
                            int z = n * columnCount + m;
                            referenceSentences.put(z, row);
                        }
                    }
                }
            }
            return r;
        }
    }

    public Matrix getFilteredCooccurrence() {
        return filteredMatrix;
    }

    public Dictionary getRowDict() {
        return rowDict;
    }

    public Dictionary getColDict() {
        return colDict;
    }

    public Multimap<Integer, Integer> getReferenceSentences() {
        return referenceSentences;
    }

    private static Options parseOptions(String[] args) {
        Options opts = new Options();
        CmdLineParser parser = new CmdLineParser(opts);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            e.printStackTrace(System.err);
            parser.printUsage(System.err);
            System.exit(1);
        }
        return opts;
    }

    private static class Options {
        @Argument
        private List<String> files = Lists.newArrayList();

        @Option(name = "-maxRowCount", usage = "Downsample rows to this size.")
        int maxRowCount = 500;

        @Option(name = "-maxColumnCount", usage = "Downsample rows to this size.")
        int maxColumnCount = 500;

        @Option(name = "-maxRelated", usage = "Maximum number of related items to be retained")
        int maxRelated = 100;

        @Override
        public String toString() {
            return "Options{" +
                    "maxColumnCount=" + maxColumnCount +
                    ", maxRowCount=" + maxRowCount +
                    ", maxRelated=" + maxRelated +
                    '}';
        }
    }

}
