package com.tdunning.cooc;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import junit.framework.Assert;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.Iterator;
import java.util.Random;

import static junit.framework.Assert.assertEquals;

public class AnalyzeTest {
    @Test
    public void testReadOccurrenceMatrix() throws IOException {
        Analyze analyzer = new Analyze();
        Matrix m = analyzer.readOccurrenceMatrix(Resources.newReaderSupplier(Resources.getResource("test-data"), Charsets.UTF_8), 1000.0, 1000.0);
        Matrix ref = new DenseMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0},
                new double[]{0, 1, 0, 1}
        });
        assertEquals(0.0, m.minus(ref).aggregate(Functions.PLUS, Functions.ABS), 0);

        Matrix cooc = ref.transpose().times(ref);
        cooc.viewDiagonal().assign(0);

        Matrix square = analyzer.square(m);
        assertEquals(0.0, cooc.minus(square).aggregate(Functions.PLUS, Functions.ABS), 0);
    }

    @Test
    public void testSampling() throws IOException {
        Vector v = new DenseVector(200);
        for (int i = 0; i < 200; i++) {
            v.set(i, 10.0 / (i + 10.0));
        }

        Matrix p = v.cross(v);
        Matrix m = p.clone().assign(new DoubleFunction() {
            Random rand = RandomUtils.getRandom();

            @Override
            public double apply(double p) {
                return rand.nextDouble() < p ? 1 : 0;
            }
        });

        m.viewDiagonal().assign(1);

        File f = Files.createTempFile("data", ".csv").toFile();
        try (PrintWriter out = new PrintWriter(f)) {
            for (MatrixSlice row : m) {
                Iterator<Vector.Element> i = row.vector().iterateNonZero();
                while (i.hasNext()) {
                    Vector.Element element = i.next();
                    out.printf("x-%d\ty-%d\n", row.index(), element.index());
                }
            }
        }


        Analyze analyzer = new Analyze();
        Matrix x = analyzer.readOccurrenceMatrix(com.google.common.io.Files.newReaderSupplier(f, Charsets.UTF_8), 1000, 1000);
        assertEquals(m.zSum(), x.zSum());

        x = analyzer.readOccurrenceMatrix(com.google.common.io.Files.newReaderSupplier(f, Charsets.UTF_8), 20, 50);
        Assert.assertTrue(rowSums(x).aggregate(Functions.MAX, Functions.IDENTITY) < 30);

        Assert.assertTrue(columnSums(x).aggregate(Functions.MAX, Functions.IDENTITY) < 65);

    }

    private Vector rowSums(Matrix x) {
        return x.aggregateRows(new VectorFunction() {
            @Override
            public double apply(Vector f) {
                return f.zSum();
            }
        });
    }

    private Vector columnSums(Matrix x) {
        return x.aggregateColumns(new VectorFunction() {
            @Override
            public double apply(Vector f) {
                return f.zSum();
            }
        });
    }


}
