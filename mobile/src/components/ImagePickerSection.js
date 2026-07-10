import React from "react";
import { View, StyleSheet, TouchableOpacity } from "react-native";
import { Card, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors } from "../constants/colors";

export default function ImagePickerSection({ onTakePhoto, onPickGallery }) {
  return (
    <Card.Content style={styles.inputSection}>
      <View style={styles.iconRow}>
        <TouchableOpacity
          onPress={onTakePhoto}
          activeOpacity={0.7}
          style={styles.iconWrapper}
        >
          <View style={[styles.iconCircle, { backgroundColor: colors.accent }]}>
            <MaterialCommunityIcons name="camera" size={36} color={colors.surface} />
          </View>
          <Text style={styles.iconLabel}>Camera</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={onPickGallery}
          activeOpacity={0.7}
          style={styles.iconWrapper}
        >
          <View style={[styles.iconCircle, { backgroundColor: colors.primaryLight }]}>
            <MaterialCommunityIcons
              name="image-multiple-outline"
              size={36}
              color={colors.surface}
            />
          </View>
          <Text style={styles.iconLabel}>Gallery</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.dropzoneText}>
        Take a photo or choose from gallery
      </Text>

      <View style={styles.tipsBox}>
        <MaterialCommunityIcons
          name="lightbulb-outline"
          size={18}
          color={colors.accent}
          style={{ marginBottom: 6 }}
        />
        <Text style={styles.tipsTitle}>Tips for Good Photos:</Text>
        <View style={styles.tipItem}>
          <MaterialCommunityIcons name="check-circle-outline" size={14} color={colors.primaryLight} />
          <Text style={styles.tipText}> Capture the entire potato leaf in focus</Text>
        </View>
        <View style={styles.tipItem}>
          <MaterialCommunityIcons name="check-circle-outline" size={14} color={colors.primaryLight} />
          <Text style={styles.tipText}> Use good, natural lighting</Text>
        </View>
        <View style={styles.tipItem}>
          <MaterialCommunityIcons name="check-circle-outline" size={14} color={colors.primaryLight} />
          <Text style={styles.tipText}> Place on a plain, neutral background</Text>
        </View>
        <View style={styles.tipItem}>
          <MaterialCommunityIcons name="check-circle-outline" size={14} color={colors.primaryLight} />
          <Text style={styles.tipText}> Capture both healthy and diseased parts</Text>
        </View>
      </View>
    </Card.Content>
  );
}

const styles = StyleSheet.create({
  inputSection: {
    alignItems: "center",
    paddingVertical: 24,
  },
  iconRow: {
    flexDirection: "row",
    justifyContent: "center",
    gap: 40,
    marginBottom: 16,
  },
  iconWrapper: {
    alignItems: "center",
  },
  iconCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: "center",
    alignItems: "center",
    elevation: 4,
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
  },
  iconLabel: {
    marginTop: 8,
    fontSize: 14,
    fontWeight: "600",
    color: colors.textPrimary,
  },
  dropzoneText: {
    color: colors.textSecondary,
    textAlign: "center",
    marginBottom: 16,
    fontSize: 14,
  },
  tipsBox: {
    backgroundColor: colors.dropzoneBg,
    borderRadius: 12,
    padding: 16,
    width: "100%",
    borderWidth: 1,
    borderColor: colors.dropzoneBorder,
  },
  tipsTitle: {
    fontWeight: "bold",
    color: colors.textPrimary,
    marginBottom: 8,
    fontSize: 14,
  },
  tipItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 2,
  },
  tipText: {
    color: "#555",
    fontSize: 13,
    flex: 1,
  },
});
