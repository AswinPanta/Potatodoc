import React, { useState, useEffect, useRef } from "react";
import { StyleSheet, ScrollView, Platform, View } from "react-native";
import { Button, Card, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { SafeAreaView } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import { colors } from "../constants/colors";
import { useModels, usePrediction, useWakeUp, MODEL_IDS, fetchGradcam } from "../hooks/useApi";
import { useHistory } from "../hooks/useHistory";
import Header from "../components/Header";
import ModelPicker from "../components/ModelPicker";
import ImagePickerSection from "../components/ImagePickerSection";
import LoadingIndicator from "../components/LoadingIndicator";
import PredictionResult from "../components/PredictionResult";
import HistoryModal from "../components/HistoryModal";

export default function HomeScreen() {
  const { awake, warming, loadingModels, statusText } = useWakeUp();
  const { models, modelNames } = useModels();
  const { data, setData, isLoading, sendFile } = usePrediction();
  const { history, saveToHistory } = useHistory();

  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUri, setImageUri] = useState(null);
  const [hasImage, setHasImage] = useState(false);
  const [selectedModel, setSelectedModel] = useState("ensemble");
  const [historyVisible, setHistoryVisible] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [heatmaps, setHeatmaps] = useState(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const fileRef = useRef(null);

  // Store file for Grad-CAM
  useEffect(() => {
    fileRef.current = selectedFile;
  }, [selectedFile]);

  useEffect(() => {
    (async () => {
      if (Platform.OS !== "web") {
        const { status: cam } =
          await ImagePicker.requestCameraPermissionsAsync();
        const { status: lib } =
          await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (cam !== "granted" || lib !== "granted") {
          console.warn("Camera or library permission denied");
        }
      }
    })();
  }, []);

  // Auto-predict when a new image is selected or model changes (no auto-save)
  useEffect(() => {
    if (!hasImage || !selectedFile || processing) return;
    const doPrediction = async () => {
      setProcessing(true);
      setHeatmaps(null);
      setSaved(false);
      await sendFile(selectedFile, selectedModel);
      setProcessing(false);
    };
    doPrediction();
  }, [imageUri, selectedModel]);

  // Fetch Grad-CAM heatmap when prediction data is available
  useEffect(() => {
    if (!data || data.error || data.is_unknown) {
      setHeatmaps(null);
      return;
    }
    const doGradcam = async () => {
      setHeatmapLoading(true);
      const result = await fetchGradcam(fileRef.current, selectedModel);
      if (result) {
        setHeatmaps(result);
      }
      setHeatmapLoading(false);
    };
    doGradcam();
  }, [data]);

  const pickFromGallery = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
    });
    if (!result.canceled && result.assets && result.assets.length > 0) {
      const asset = result.assets[0];
      setSelectedFile(asset);
      setImageUri(asset.uri);
      setHasImage(true);
      setData(null);
    }
  };

  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      quality: 1,
    });
    if (!result.canceled && result.assets && result.assets.length > 0) {
      const asset = result.assets[0];
      setSelectedFile(asset);
      setImageUri(asset.uri);
      setHasImage(true);
      setData(null);
    }
  };

  const nextImage = () => {
    setData(null);
    setHasImage(false);
    setSelectedFile(null);
    setImageUri(null);
    setHeatmaps(null);
    setSaved(false);
  };

  const handleSaveToHistory = () => {
    saveToHistory({
      ...data,
      model:
        selectedModel === "ensemble"
          ? "Ensemble"
          : modelNames[selectedModel],
      heatmap: heatmaps,
    });
    setSaved(true);
  };

  const handleModelChange = (value) => {
    setSelectedModel(value);
    if (data) {
      setData(null);
    }
    setHeatmaps(null);
    setSaved(false);
  };

  const hasSaveableResult = data && !data.error && data.class !== "Unknown";

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <Header onHistoryPress={() => setHistoryVisible(true)} />

      {warming && (
        <View style={styles.wakeBanner}>
          <MaterialCommunityIcons
            name={awake ? "lightning-bolt" : "wifi"}
            size={14}
            color={colors.surface}
          />
          <Text style={styles.wakeText}>{statusText}</Text>
          {loadingModels.length > 0 && (
            <Text style={styles.wakeModels}>
              {loadingModels.length}/{MODEL_IDS.length}
            </Text>
          )}
        </View>
      )}

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
      >
        <ModelPicker
          models={models}
          modelNames={modelNames}
          selectedModel={selectedModel}
          onChange={handleModelChange}
        />

        <Card style={styles.card}>
          {hasImage && imageUri && (
            <Card.Cover
              source={{ uri: imageUri }}
              style={styles.imagePreview}
            />
          )}

          {!hasImage && (
            <ImagePickerSection
              onTakePhoto={takePhoto}
              onPickGallery={pickFromGallery}
            />
          )}

          {isLoading && <LoadingIndicator />}

          {data && (
            <PredictionResult
              data={data}
              heatmaps={heatmaps}
              heatmapLoading={heatmapLoading}
              selectedModel={selectedModel}
              modelNames={modelNames}
            />
          )}
        </Card>

        {data && (
          <View style={styles.actionContainer}>
            <Button
              mode="contained"
              onPress={nextImage}
              style={styles.actionButton}
              buttonColor={colors.accent}
              labelStyle={styles.actionButtonLabel}
              icon={() => (
                <MaterialCommunityIcons name="camera" size={20} color="white" />
              )}
            >
              Next Image
            </Button>

            {hasSaveableResult && (
              <Button
                mode="contained"
                onPress={handleSaveToHistory}
                style={styles.actionButton}
                buttonColor={saved ? colors.primaryLight : colors.primary}
                labelStyle={styles.actionButtonLabel}
                disabled={saved}
                icon={() => (
                  <MaterialCommunityIcons
                    name={saved ? "check" : "content-save"}
                    size={20}
                    color="white"
                  />
                )}
              >
                {saved ? "Saved!" : "Save to History"}
              </Button>
            )}
          </View>
        )}
      </ScrollView>

      <HistoryModal
        visible={historyVisible}
        onClose={() => setHistoryVisible(false)}
        history={history}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    alignItems: "center",
  },
  card: {
    width: "100%",
    maxWidth: 480,
    borderRadius: 20,
    elevation: 4,
    marginTop: 8,
    overflow: "hidden",
  },
  imagePreview: {
    height: 300,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  actionContainer: {
    width: "100%",
    maxWidth: 480,
    marginTop: 16,
    gap: 12,
  },
  actionButton: {
    width: "100%",
    borderRadius: 12,
    paddingVertical: 4,
  },
  actionButtonLabel: {
    fontSize: 16,
    fontWeight: "bold",
    paddingVertical: 4,
  },
  wakeBanner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.primaryLight,
    paddingVertical: 6,
    paddingHorizontal: 16,
    gap: 6,
  },
  wakeText: {
    color: colors.surface,
    fontSize: 13,
    fontWeight: "500",
  },
  wakeModels: {
    color: colors.surface,
    fontSize: 11,
    fontWeight: "bold",
    backgroundColor: "rgba(255,255,255,0.25)",
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 2,
    overflow: "hidden",
  },
});
