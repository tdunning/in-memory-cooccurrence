package com.tdunning.cooc;

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Ordering;
import com.google.common.io.CharStreams;
import com.google.common.io.Files;
import com.google.common.io.InputSupplier;
import com.google.common.io.LineProcessor;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.LogLikelihood;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Analyzes data in a sparse matrix for interesting cooccurrence.
 *
 * The input is assumed to contain two tab separated fields which are string names
 * of the row and column of the cooccurrence matrix, respectively.  The output
 * of this method is a reduced cooccurrence matrix suitable for indexing.
 *
 * The process of reduction involves:
 *
 * <ul>
 *     <li>downsampling the data to limit row and column cardinality</li>
 *     <li>using LLR to find anomalous cooccurrence</li>
 * </ul>
 *
 * This is done by making several passes over an input file.  The first pass is used
 * to create dictionaries of the symbols involved.  The second pass populates the
 * cooccurrence matrix while down-sampling the data.
 */
public class Analyze {
    private static final int ROW_LIMIT_SIZE = 200;
    public static final Splitter onTab = Splitter.on("\t");
    private final Dictionary rowDict;
    private final Dictionary colDict;
    private final Matrix filteredMatrix;

    /**
     * This is for illustration purposes only at this time.
     *
     * @param args  A file to process
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        Analyze analyzer = new Analyze(Files.newReaderSupplier(new File(args[0]), Charsets.UTF_8));
        Matrix m = analyzer.getFilteredCooccurrence();
        Dictionary colDict = analyzer.getColDict();
        for (MatrixSlice row : m) {
            Iterator<Vector.Element> i = row.vector().iterateNonZero();
            while (i.hasNext()) {
                Vector.Element element = i.next();
                System.out.printf("%s\t%s\n", colDict.values().get(row.index()), colDict.values().get(element.index()));
            }
        }
    }

    /**
     * Analyze an input to find significant cooccurrences.  The input should consist of tab-delimited
     * row and column designators.  The row is the unit of text for textual cooccurrence or the user
     * for recommendations.  The output will be a matrix of significant cooccurrences.
     * @throws IOException
     * @param input  A tab-delimited input file with row and column descriptors
     */
    public Analyze(InputSupplier<InputStreamReader> input) throws IOException {
        rowDict = new Dictionary();
        colDict = new Dictionary();

        final Multiset<Integer> rowCounts = HashMultiset.create();
        final Multiset<Integer> colCounts = HashMultiset.create();

        // read the data and build dictionaries.  This tells us how large the occurrence matrix must be
        CharStreams.readLines(input, new LineProcessor<Object>() {

            public boolean processLine(String s) throws IOException {
                Iterator<String> x = onTab.split(s).iterator();
                rowCounts.add(rowDict.intern(x.next()));
                colCounts.add(colDict.intern(x.next()));
                return true;
            }

            public Object getResult() {
                return null;
            }
        });


        // now we can read the actual data.  Note that we downsample this data based on our first pass
        Matrix occurrences = CharStreams.readLines(input, new LineProcessor<Matrix>() {
            Matrix m = new SparseRowMatrix(rowDict.size(), colDict.size(), true);

            public boolean processLine(String s) throws IOException {
                Iterator<String> x = onTab.split(s).iterator();
                int row = rowDict.intern(x.next());
                int col = colDict.intern(x.next());

                double rowRate = Math.min(1000.0, rowCounts.count(row)) / rowCounts.count(row);
                double colRate = Math.min(1000.0, colCounts.count(col)) / colCounts.count(col);
                Random random = RandomUtils.getRandom();
                // this down-samples by the product of both factors.  Almost always, one factor will be == 1.
                // if that assumption turns out wrong, we might down-sample according to Math.min(rowRate, colRate)
                if (random.nextDouble() < rowRate && random.nextDouble() < colRate) {
                    m.set(row, col, 1);
                }
                return true;
            }

            public Matrix getResult() {
                return m;
            }
        });

        // now we square the occurrence matrix to get cooccurrences
        Matrix cooccurrence = square(occurrences);

        // to determine anomalous cooccurrence, we need row and column sums
        Vector rowSums = new DenseVector(rowDict.size());
        Vector colSums = new DenseVector(colDict.size());
        for (MatrixSlice row : cooccurrence) {
            rowSums.set(row.index(), row.vector().zSum());
        }

        for (int i = 0; i < colDict.size(); i++) {
            colSums.set(i, cooccurrence.viewColumn(i).zSum());
        }

        // and the total
        double total = rowSums.zSum();

        for (MatrixSlice row : cooccurrence) {
            Iterator<Vector.Element> elements = row.vector().iterateNonZero();
            while (elements.hasNext()) {
                Vector.Element element = elements.next();
                long k11 = (long) element.get();
                long k12 = (long) (rowSums.get(row.index()) - k11);
                long k21 = (long) (colSums.get(element.index()) - k11);
                long k22 = (long) (total - k11 - k12 - k21);
                double score = LogLikelihood.rootLogLikelihoodRatio(k11, k12, k21, k22);
                element.set(score);
            }
        }

        filteredMatrix = new SparseRowMatrix(rowDict.size(), colDict.size(), true);
        for (MatrixSlice row : cooccurrence) {
            List<Vector.Element> elements = Lists.newArrayList(row.vector().iterateNonZero());
            Collections.sort(elements, new Ordering<Vector.Element>() {
                public int compare(Vector.Element o1, Vector.Element o2) {
                    return Double.compare(o1.get(), o2.get());
                }
            }.reverse());
            elements = elements.subList(0, Math.min(ROW_LIMIT_SIZE, elements.size()));
            for (Vector.Element element : elements) {
                filteredMatrix.set(row.index(), element.index(), 1);
            }
        }

    }

    /**
     * Given a matrix A, return A' * A which can be interpreted as a cooccurrence matrix.
     * @param occurrences The original occurrence data
     * @return The cooccurrence data.
     */
    private Matrix square(Matrix occurrences) {
        Matrix r = new SparseRowMatrix(occurrences.columnSize(), occurrences.columnSize());
        for (MatrixSlice row : occurrences) {
            Iterator<Vector.Element> i = row.vector().iterateNonZero();
            Iterator<Vector.Element> j = row.vector().iterateNonZero();
            while (i.hasNext()) {
                Vector.Element e1 = i.next();
                while (j.hasNext()) {
                    Vector.Element e2 = j.next();
                    if (e1.index() != e2.index()) {
                        r.set(e1.index(), e2.index(), r.get(e1.index(), e2.index()) + e1.get() * e2.get());
                    }
                }
            }
        }
        return r;
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
}
