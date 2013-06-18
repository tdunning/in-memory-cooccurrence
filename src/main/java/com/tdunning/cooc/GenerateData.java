package com.tdunning.cooc;

import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnel;
import com.google.common.hash.PrimitiveSink;
import org.apache.mahout.math.random.ChineseRestaurant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Generate somewhat realistic data files for cooccurrence analysis.
 */
public class GenerateData {
    private static Logger log = LoggerFactory.getLogger(GenerateData.class);

    public static void main(String[] args) throws IOException {
        for (String arg : args) {
            generate(Integer.parseInt(arg));
        }
    }

    private static void generate(int scale) throws IOException {
        ChineseRestaurant row = new ChineseRestaurant(20000, 0.35);
        ChineseRestaurant col = new ChineseRestaurant(20000, 0.35);
        int n = (int) Math.pow(10, scale);

        // we use a bloom filter to avoid generating duplicates
        Funnel<Long> f = new Funnel<Long>() {
            @Override
            public void funnel(Long from, PrimitiveSink into) {
                into.putLong(from);
            }
        };
        BloomFilter<Long> unique = BloomFilter.create(f, 10 * n, 0.1 / n);

        File dataFile = new File(String.format("data-%d.tsv", scale));
        log.info("Generating data into {}", dataFile);
        PrintWriter out = new PrintWriter(dataFile);
        LineCounter counter = new LineCounter(log, "   Generating");
        for (int k = 0; k < n; k++) {
            int i = row.sample();
            int j = col.sample();
            long combo = ((long) i << 32) + j;
            if (!unique.mightContain(combo)) {
                unique.put(combo);
                out.printf("x-%d\ty-%d\n", i, j);
            }
            counter.step();
        }
        out.close();
        log.info("Done generating {} x {}", row.size(), col.size());

    }
}
