package com.tdunning.cooc;

import com.google.common.collect.ImmutableMap;
import org.slf4j.Logger;

import java.util.Map;

/**
* Log progress at an exponentially decreasing rate.
*/
class LineCounter {
    private int n = 0;
    private int step = 1000;
    private String message;
    Logger log;

    LineCounter(Logger log) {
        this(log, "Line");
    }

    LineCounter(Logger log, String message) {
        this.log = log;
        this.message = message + " {}";
    }

    public void step() {
        if (n % step == 0) {
            log.info(message, unitize(n));
            if (n / step >= 5) {
                step = step * 10;
            }
        }
        n++;
    }

    private String unitize(int n) {
        Map<Double, String> units = ImmutableMap.of(1e9, "G", 1e6, "M", 1e3, "K");
        for (Double scale : units.keySet()) {
            if (n >= scale) {
                return String.format("%.0f %s", n / scale, units.get(scale));
            }
        }
        return String.format("%d", n);
    }
}
