import React, { useState } from "react";
import { View, Image, StyleSheet, TouchableOpacity, ActivityIndicator } from "react-native";
import {
  Text,
  Card,
  Title,
  Paragraph,
  Divider,
} from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors } from "../constants/colors";
import DISEASE_INFO from "../constants/diseaseInfo";

const CLASS_COLORS = {
  "Early Blight": "#C62828",
  "Late Blight": "#E65100",
  "Healthy": "#2E7D32",
};

function ProbabilityBarChart({ probabilities, predictedClass }) {
  if (!probabilities) return null;

  const entries = Object.entries(probabilities);

  return (
    <View style={styles.probabilitySection}>
      <Text style={styles.probabilityTitle}>Per-Class Probabilities</Text>
      {entries.map(([className, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const isPredicted = className === predictedClass;
        const barColor = CLASS_COLORS[className] || "#666";
        return (
          <View key={className} style={styles.probabilityRow}>
            <View style={styles.probabilityLabelRow}>
              <Text
                style={[
                  styles.probabilityLabel,
                  isPredicted && styles.probabilityLabelActive,
                  isPredicted && { color: barColor },
                ]}
              >
                {className} {isPredicted ? "←" : ""}
              </Text>
              <Text
                style={[
                  styles.probabilityValue,
                  isPredicted && styles.probabilityLabelActive,
                  isPredicted && { color: barColor },
                ]}
              >
                {pct}%
              </Text>
            </View>
            <View style={styles.probabilityBarBg}>
              <View
                style={[
                  styles.probabilityBarFill,
                  {
                    width: `${pct}%`,
                    backgroundColor: barColor,
                    opacity: isPredicted ? 1 : 0.4,
                  },
                ]}
              />
            </View>
          </View>
        );
      })}
    </View>
  );
}

function DiseaseInfoCard({ diseaseClass }) {
  const info = DISEASE_INFO[diseaseClass];
  if (!info) return null;

  return (
    <View style={styles.infoSection}>
      <Title style={styles.infoTitle}>About {diseaseClass}</Title>
      <Paragraph style={styles.description}>{info.description}</Paragraph>

      <Text style={styles.sectionLabel}>Symptoms:</Text>
      {info.symptoms.map((s, i) => (
        <Text key={i} style={styles.listItem}>
          • {s}
        </Text>
      ))}

      <Text style={[styles.sectionLabel, { marginTop: 12 }]}>
        {diseaseClass === "Healthy" ? "Care Tips:" : "Treatment & Mitigation:"}
      </Text>
      {info.treatment.map((t, i) => (
        <Text key={i} style={styles.listItem}>
          • {t}
        </Text>
      ))}
    </View>
  );
}

function HeatmapSection({ heatmaps, heatmapLoading, selectedModel, modelNames }) {
  const [showHeatmap, setShowHeatmap] = useState(false);

  if (heatmapLoading) {
    return (
      <View style={styles.heatmapSection}>
        <View style={styles.heatmapLoadingRow}>
          <ActivityIndicator size={16} color={colors.primary} />
          <Text style={styles.heatmapLoadingText}>
            Computing heatmap...
          </Text>
        </View>
      </View>
    );
  }

  if (!heatmaps) return null;

  const hasEnsemble = selectedModel === "ensemble";

  return (
    <View style={styles.heatmapSection}>
      <TouchableOpacity
        style={styles.heatmapHeader}
        onPress={() => setShowHeatmap(!showHeatmap)}
        activeOpacity={0.7}
      >
        <View style={styles.heatmapHeaderLeft}>
          <MaterialCommunityIcons
            name="fire"
            size={20}
            color="#E65100"
          />
          <Text style={styles.heatmapHeaderText}>
            Explainable AI - Heatmap
          </Text>
        </View>
        <MaterialCommunityIcons
          name={showHeatmap ? "chevron-up" : "chevron-down"}
          size={24}
          color={colors.textPrimary}
        />
      </TouchableOpacity>

      {showHeatmap && (
        <View style={styles.heatmapContent}>
          <Text style={styles.heatmapDescription}>
            The heatmap highlights which regions of the image the model
            focused on to make its prediction. Red areas indicate high
            importance, blue areas low importance.
          </Text>

          {hasEnsemble && heatmaps.heatmaps ? (
            <View style={styles.heatmapList}>
              {Object.entries(heatmaps.heatmaps).map(
                ([modelName, modelData]) => (
                  <View key={modelName} style={styles.heatmapItem}>
                    <Text style={styles.heatmapModelLabel}>
                      {modelName}
                    </Text>
                    {modelData && modelData.overlay ? (
                      <Image
                        source={{ uri: modelData.overlay }}
                        style={styles.heatmapImage}
                        resizeMode="contain"
                      />
                    ) : (
                      <Text style={styles.heatmapUnavailable}>
                        Heatmap unavailable for this model
                      </Text>
                    )}
                  </View>
                )
              )}
            </View>
          ) : (
            (heatmaps.overlay || heatmaps.heatmap) && (
              <Image
                source={{ uri: heatmaps.overlay || heatmaps.heatmap }}
                style={styles.heatmapImage}
                resizeMode="contain"
              />
            )
          )}

          <Text style={styles.heatmapCaption}>
            Grad-CAM visualization • Red = high importance
          </Text>
        </View>
      )}
    </View>
  );
}

function UnknownImageCard({ data }) {
  const confidence = (parseFloat(data.confidence) * 100).toFixed(2);

  return (
    <View style={styles.unknownCard}>
      <MaterialCommunityIcons
        name="alert-circle-outline"
        size={48}
        color="#E65100"
        style={{ marginBottom: 12 }}
      />
      <Title style={styles.unknownTitle}>Unknown Image</Title>
      <Text style={styles.unknownText}>
        {data.message ||
          "This does not appear to be a potato leaf."}
      </Text>
      <Text style={styles.unknownConfidence}>
        Max confidence: {confidence}%
      </Text>

      <View style={styles.unknownSuggestions}>
        <Text style={styles.unknownSuggestionTitle}>
          Suggestions:
        </Text>
        <View style={styles.suggestionItem}>
          <MaterialCommunityIcons name="circle-small" size={16} color="#E65100" />
          <Text style={styles.suggestionText}>Upload a clear photo of a potato leaf</Text>
        </View>
        <View style={styles.suggestionItem}>
          <MaterialCommunityIcons name="circle-small" size={16} color="#E65100" />
          <Text style={styles.suggestionText}>Ensure the leaf occupies most of the frame</Text>
        </View>
        <View style={styles.suggestionItem}>
          <MaterialCommunityIcons name="circle-small" size={16} color="#E65100" />
          <Text style={styles.suggestionText}>Use good lighting for better results</Text>
        </View>
        <View style={styles.suggestionItem}>
          <MaterialCommunityIcons name="circle-small" size={16} color="#E65100" />
          <Text style={styles.suggestionText}>Avoid non-plant or blurred images</Text>
        </View>
      </View>
    </View>
  );
}

export default function PredictionResult({
  data,
  heatmaps,
  heatmapLoading,
  selectedModel,
  modelNames,
}) {
  // Handle unknown image
  if (data.is_unknown || data.class === "Unknown") {
    return (
      <Card.Content style={styles.container}>
        <UnknownImageCard data={data} />
      </Card.Content>
    );
  }

  const confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  const isLowConfidence = parseFloat(confidence) < 70;
  const isHealthy = data.class === "Healthy";
  const resultColor = isHealthy ? colors.primaryDark : "#B71C1C";

  return (
    <Card.Content style={styles.container}>
      <Title style={[styles.heading, { color: resultColor }]}>
        Prediction Result
      </Title>

      {/* Main Prediction Card */}
      <Card style={[styles.resultCard, { borderLeftColor: resultColor, borderLeftWidth: 4 }]}>
        <Card.Content>
          <View style={styles.resultRow}>
            <Text style={styles.label}>Label</Text>
            <View
              style={[
                styles.chip,
                {
                  backgroundColor: isHealthy ? "#E8F5E9" : "#FFEBEE",
                },
              ]}
            >
              <Text style={[styles.chipText, { color: resultColor }]}>
                {data.class}
              </Text>
            </View>
          </View>
          <Divider style={styles.divider} />
          <View style={styles.resultRow}>
            <Text style={styles.label}>Confidence</Text>
            <Text style={[styles.value, { color: resultColor }]}>
              {confidence}%
            </Text>
          </View>
        </Card.Content>
      </Card>

      {/* Per-Class Probabilities Bar Chart */}
      {data.probabilities && (
        <ProbabilityBarChart
          probabilities={data.probabilities}
          predictedClass={data.class}
        />
      )}

      {/* Low Confidence Warning */}
      {isLowConfidence && (
        <View style={styles.warningBox}>
          <View style={styles.warningRow}>
            <MaterialCommunityIcons
              name="alert"
              size={18}
              color={colors.warningText}
            />
            <Text style={styles.warningTitle}>
              Low Confidence Prediction
            </Text>
          </View>
          <Text style={styles.warningText}>
            The model is not very confident. Try retaking the photo
            with better lighting and a clearer leaf image.
          </Text>
        </View>
      )}

      {/* Individual Model Predictions (Ensemble) */}
      {data.individual && data.individual.length > 0 && (
        <View style={styles.individualSection}>
          <Title style={styles.subheading}>
            Individual Model Predictions
          </Title>
          <Card style={styles.resultCard}>
            <Card.Content>
              <View style={styles.tableHeader}>
                <Text style={[styles.label, styles.colModel]}>Model</Text>
                <Text style={[styles.label, styles.colClass]}>Label</Text>
                <Text style={[styles.label, styles.colConf]}>
                  Confidence
                </Text>
              </View>
              <Divider style={styles.divider} />
              {data.individual.map((pred, idx) => (
                <View key={idx}>
                  <View style={styles.tableRow}>
                    <Text style={[styles.tableCell, styles.colModel]}>
                      {pred.model}
                    </Text>
                    <Text style={[styles.tableCell, styles.colClass]}>
                      {pred.class}
                    </Text>
                    <Text style={[styles.tableCell, styles.colConf]}>
                      {(pred.confidence * 100).toFixed(2)}%
                    </Text>
                  </View>
                  {idx < data.individual.length - 1 && (
                    <Divider style={styles.divider} />
                  )}
                </View>
              ))}
            </Card.Content>
          </Card>
        </View>
      )}

      {/* Grad-CAM Heatmap Section */}
      <HeatmapSection
        heatmaps={heatmaps}
        heatmapLoading={heatmapLoading}
        selectedModel={selectedModel}
        modelNames={modelNames}
      />

      {/* Disease Info */}
      <DiseaseInfoCard diseaseClass={data.class} />
    </Card.Content>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
  },
  heading: {
    textAlign: "center",
    fontWeight: "bold",
    marginBottom: 16,
  },
  subheading: {
    color: colors.textPrimary,
    fontWeight: "bold",
    marginBottom: 8,
    fontSize: 18,
  },
  resultCard: {
    borderRadius: 12,
    elevation: 2,
    marginBottom: 12,
  },
  resultRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 8,
  },
  label: {
    fontSize: 16,
    color: colors.textPrimary,
    fontWeight: "600",
  },
  value: {
    fontSize: 20,
    fontWeight: "bold",
  },
  chip: {
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  chipText: {
    fontWeight: "bold",
    fontSize: 16,
  },
  divider: {
    backgroundColor: colors.border,
  },
  warningBox: {
    backgroundColor: colors.warning,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#FFEEBA",
  },
  warningRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    marginBottom: 6,
  },
  warningTitle: {
    color: colors.warningText,
    fontWeight: "bold",
    fontSize: 15,
  },
  warningText: {
    color: colors.warningText,
    fontSize: 14,
  },
  individualSection: {
    marginTop: 8,
    marginBottom: 16,
  },
  tableHeader: {
    flexDirection: "row",
    paddingVertical: 8,
  },
  tableRow: {
    flexDirection: "row",
    paddingVertical: 10,
  },
  colModel: { flex: 2 },
  colClass: { flex: 2 },
  colConf: { flex: 1.5, textAlign: "right" },
  tableCell: {
    fontSize: 14,
    color: colors.textPrimary,
  },
  infoSection: {
    marginTop: 16,
    marginBottom: 24,
  },
  infoTitle: {
    color: colors.primary,
    fontWeight: "bold",
    marginBottom: 4,
  },
  description: {
    fontSize: 14,
    marginBottom: 12,
    color: "#333",
    lineHeight: 20,
  },
  sectionLabel: {
    fontWeight: "bold",
    color: colors.textPrimary,
    marginBottom: 4,
  },
  listItem: {
    fontSize: 14,
    color: "#333",
    paddingVertical: 2,
    paddingLeft: 4,
    lineHeight: 20,
  },
  // Heatmap styles
  heatmapSection: {
    marginTop: 16,
    marginBottom: 16,
    backgroundColor: "#FAFAFA",
    borderRadius: 12,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#E8F5E9",
  },
  heatmapLoadingRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    padding: 16,
  },
  heatmapLoadingText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: "500",
  },
  heatmapHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 14,
    backgroundColor: "#F1F8E9",
    borderBottomWidth: 1,
    borderBottomColor: "#E8F5E9",
  },
  heatmapHeaderLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  heatmapHeaderText: {
    color: "#2E7D32",
    fontWeight: "bold",
    fontSize: 15,
  },
  heatmapContent: {
    padding: 16,
  },
  heatmapDescription: {
    color: "#666",
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 12,
  },
  heatmapList: {
    gap: 16,
  },
  heatmapItem: {
    marginBottom: 4,
  },
  heatmapModelLabel: {
    color: "#555",
    fontWeight: "600",
    fontSize: 12,
    marginBottom: 6,
  },
  heatmapImage: {
    width: "100%",
    height: 200,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "#E8F5E9",
  },
  heatmapUnavailable: {
    color: "#999",
    fontSize: 12,
    fontStyle: "italic",
  },
  heatmapCaption: {
    color: "#999",
    fontSize: 11,
    textAlign: "center",
    marginTop: 8,
  },
  // Unknown image card
  unknownCard: {
    alignItems: "center",
    paddingVertical: 16,
  },
  unknownTitle: {
    color: "#E65100",
    fontWeight: "bold",
    marginBottom: 8,
  },
  unknownText: {
    color: "#555",
    textAlign: "center",
    fontSize: 15,
    marginBottom: 8,
  },
  unknownConfidence: {
    color: "#888",
    fontSize: 13,
    marginBottom: 16,
  },
  unknownSuggestions: {
    backgroundColor: "#FFF3E0",
    borderRadius: 12,
    padding: 16,
    width: "100%",
  },
  unknownSuggestionTitle: {
    color: "#E65100",
    fontWeight: "bold",
    marginBottom: 8,
    fontSize: 14,
  },
  suggestionItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 2,
  },
  suggestionText: {
    color: "#555",
    fontSize: 13,
    flex: 1,
  },
  // Probability bar chart
  probabilitySection: {
    marginTop: 16,
    marginBottom: 16,
    width: "100%",
  },
  probabilityTitle: {
    color: colors.textPrimary,
    fontWeight: "bold",
    fontSize: 16,
    marginBottom: 12,
  },
  probabilityRow: {
    marginBottom: 10,
  },
  probabilityLabelRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  probabilityLabel: {
    fontSize: 13,
    color: "#555",
  },
  probabilityLabelActive: {
    fontWeight: "bold",
  },
  probabilityValue: {
    fontSize: 13,
    color: "#555",
  },
  probabilityBarBg: {
    width: "100%",
    height: 10,
    backgroundColor: "#E8E8E8",
    borderRadius: 5,
    overflow: "hidden",
  },
  probabilityBarFill: {
    height: "100%",
    borderRadius: 5,
  },
});
