package com.tdunning.cooc;

import com.google.common.base.Splitter;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.io.CharStreams;
import com.google.common.io.InputSupplier;
import com.google.common.io.LineProcessor;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.stats.LogLikelihood;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

/**
 * Analyzes data in a sparse matrix for interesting cooccurrence.
 *
 * This involves:
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
    public final Splitter onTab = Splitter.on("\t");

    public Matrix analyze(InputSupplier<BufferedReader> input) throws IOException {
        final Multiset<Integer> rowCounts = HashMultiset.create();
        final Multiset<Integer> colCounts = HashMultiset.create();

        final Dictionary rowDict = new Dictionary();
        final Dictionary colDict = new Dictionary();

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


        Matrix cooccurrence = CharStreams.readLines(input, new LineProcessor<Matrix>() {
            Matrix m = new SparseRowMatrix(rowDict.size(), colDict.size(), true);

            public boolean processLine(String s) throws IOException {
                Iterator<String> x = onTab.split(s).iterator();
                int row = rowDict.intern(x.next());
                int col = colDict.intern(x.next());

                double rowRate = Math.min(1000.0, rowCounts.count(row)) / rowCounts.count(row);
                double colRate = Math.min(1000.0, colCounts.count(col)) / colCounts.count(col);
                Random random = RandomUtils.getRandom();
                if (random.nextDouble() < rowRate && random.nextDouble() < colRate) {
                    m.set(row, col, m.get(row, col) + 1);
                }
                return true;
            }

            public Matrix getResult() {
                return m;
            }
        });

        Vector rowSums = new DenseVector(rowDict.size());
        Vector colSums = new DenseVector(colDict.size());
        for (MatrixSlice row : cooccurrence) {
            rowSums.set(row.index(), row.vector().zSum());
        }

        double total = rowSums.zSum();

        for (int i = 0; i < colDict.size(); i++) {
            colSums.set(i, cooccurrence.viewColumn(i).zSum());
        }

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

        return cooccurrence;
    }
}
