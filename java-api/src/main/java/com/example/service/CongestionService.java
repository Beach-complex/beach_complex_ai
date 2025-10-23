package com.example.service;

import com.example.ml.OnnxPredictor;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class CongestionService {
    private static final List<String> FEATURES = List.of(
            "feature_temp",
            "feature_rain",
            "feature_is_holiday"
    );
    private static final String[] LABELS = {"Green", "Yellow", "Red"};

    private final WebClient prophetClient = WebClient.builder()
            .baseUrl("http://prophet-infer:8000")
            .build();

    private OnnxPredictor predictor;

    @PostConstruct
    public void init() throws Exception {
        this.predictor = new OnnxPredictor("/app/models/onnx/active.onnx", FEATURES);
    }

    public Map<String, Object> classify(Map<String, Number> features) throws Exception {
        Map<String, Float> floatFeatures = features.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> entry.getValue() == null ? 0.0f : entry.getValue().floatValue()
                ));

        float[] probabilities = predictor.predictProba(floatFeatures);
        int labelIndex = 0;
        float max = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > max) {
                max = probabilities[i];
                labelIndex = i;
            }
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("probs", probabilities);
        response.put("label", LABELS[labelIndex]);
        return response;
    }

    public Mono<Map> forecastProphet(String stationId, int horizonHours) {
        Map<String, Object> payload = Map.of(
                "station_id", stationId,
                "horizon_hours", horizonHours
        );

        return prophetClient.post()
                .uri("/forecast")
                .bodyValue(payload)
                .retrieve()
                .bodyToMono(Map.class);
    }
}
