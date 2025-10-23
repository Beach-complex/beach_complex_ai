FROM eclipse-temurin:21-jdk AS build

WORKDIR /workspace
COPY . .
RUN ./gradlew --version \
 && ./gradlew clean bootJar -x test

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /workspace/build/libs/*.jar app.jar
RUN mkdir -p /app/models/onnx
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
