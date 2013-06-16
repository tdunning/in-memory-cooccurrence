package com.tdunning.cooc;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.io.IOException;

import static junit.framework.Assert.assertEquals;

public class AnalyzeTest {
    @Test
    public void testReadOccurrenceMatrix() throws IOException {
        Analyze analyzer = new Analyze();
        Matrix m = analyzer.readOccurrenceMatrix(Resources.newReaderSupplier(Resources.getResource("test-data"), Charsets.UTF_8));
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
}
