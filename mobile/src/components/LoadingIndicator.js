import React from "react";
import { StyleSheet } from "react-native";
import { Card, Text } from "react-native-paper";
import { ActivityIndicator } from "react-native";
import { colors } from "../constants/colors";

export default function LoadingIndicator() {
  return (
    <Card.Content style={styles.loadingSection}>
      <ActivityIndicator size="large" color={colors.primary} />
      <Text style={styles.loadingText}>Processing...</Text>
    </Card.Content>
  );
}

const styles = StyleSheet.create({
  loadingSection: {
    alignItems: "center",
    paddingVertical: 32,
  },
  loadingText: {
    marginTop: 12,
    color: colors.textPrimary,
    fontWeight: "600",
    fontSize: 16,
  },
});
