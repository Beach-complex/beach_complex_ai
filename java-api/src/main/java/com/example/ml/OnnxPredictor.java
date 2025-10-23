package com.example.ml;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public final class OnnxPredictor implements Closeable {
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final List<String> features;
    private final String inputName;

    public OnnxPredictor(String modelPath, List<String> features) throws OrtException {
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(modelPath, new OrtSession.SessionOptions());
        this.features = features;
        this.inputName = session.getInputNames().iterator().next();
    }

    public float[] predictProba(Map<String, Float> featureMap) throws OrtException {
        float[] values = new float[features.size()];
        for (int i = 0; i < features.size(); i++) {
            values[i] = featureMap.getOrDefault(features.get(i), 0.0f);
        }

        long[] shape = new long[]{1, features.size()};
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), shape)) {
            try (OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                Object value = result.get(0).getValue();
                if (value instanceof float[][] probs && probs.length > 0) {
                    return probs[0];
                }
                throw new OrtException("Unexpected ONNX output type: " + value.getClass());
            }
        }
    }

    @Override
    public void close() throws IOException {
        try {
            session.close();
        } catch (OrtException e) {
            throw new IOException("Failed to close ONNX session", e);
        }
        environment.close();
    }
}
