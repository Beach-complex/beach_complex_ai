package com.example.api;

import com.example.service.CongestionService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
@RequestMapping("/v1")
public class CongestionController {
    private final CongestionService service;

    public CongestionController(CongestionService service) {
        this.service = service;
    }

    @PostMapping(value = "/classify", consumes = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> classify(@RequestBody Map<String, Number> features) throws Exception {
        return service.classify(features);
    }

    @GetMapping("/forecast")
    public Mono<Map> forecast(@RequestParam("stationId") String stationId,
                              @RequestParam(name = "hours", defaultValue = "24") int horizon) {
        return service.forecastProphet(stationId, horizon);
    }
}
