import React from "react";
import { View, FlatList, StyleSheet } from "react-native";
import { Modal, Portal, Text, Button, Divider } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors } from "../constants/colors";

function HistoryItem({ item }) {
  const isHealthy = item.class === "Healthy";
  const iconName = isHealthy ? "leaf" : "alert-circle-outline";
  const iconColor = isHealthy ? colors.primaryLight : colors.accent;

  return (
    <View style={styles.item}>
      <View style={styles.itemRow}>
        <MaterialCommunityIcons
          name={iconName}
          size={18}
          color={iconColor}
          style={styles.itemIcon}
        />
        <View style={styles.itemText}>
          <Text style={styles.itemPrimary}>
            {item.class} ({(item.confidence * 100).toFixed(2)}%)
          </Text>
          <Text style={styles.itemSecondary}>
            {item.model} · {item.timestamp}
          </Text>
        </View>
      </View>
      <Divider style={styles.divider} />
    </View>
  );
}

export default function HistoryModal({ visible, onClose, history }) {
  return (
    <Portal>
      <Modal
        visible={visible}
        onDismiss={onClose}
        contentContainerStyle={styles.modal}
      >
        <View style={styles.headerRow}>
          <MaterialCommunityIcons name="history" size={24} color={colors.primary} />
          <Text style={styles.title}>Prediction History</Text>
        </View>

        <FlatList
          data={history}
          keyExtractor={(item) => String(item.id)}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <MaterialCommunityIcons
                name="history"
                size={48}
                color={colors.border}
              />
              <Text style={styles.empty}>
                No predictions yet.
              </Text>
              <Text style={styles.emptySub}>
                Start predicting to see history here.
              </Text>
            </View>
          }
          renderItem={({ item }) => <HistoryItem item={item} />}
        />

        <Button
          mode="contained"
          onPress={onClose}
          style={styles.closeButton}
          labelStyle={styles.closeButtonLabel}
          buttonColor={colors.primary}
          icon={() => <MaterialCommunityIcons name="close" size={18} color="white" />}
        >
          Close
        </Button>
      </Modal>
    </Portal>
  );
}

const styles = StyleSheet.create({
  modal: {
    backgroundColor: "white",
    margin: 24,
    borderRadius: 16,
    padding: 24,
    maxHeight: "80%",
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
    gap: 8,
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    color: colors.textPrimary,
  },
  emptyContainer: {
    alignItems: "center",
    padding: 32,
  },
  empty: {
    textAlign: "center",
    color: colors.textSecondary,
    marginTop: 12,
    fontSize: 16,
    fontWeight: "500",
  },
  emptySub: {
    textAlign: "center",
    color: colors.textSecondary,
    marginTop: 4,
    fontSize: 13,
  },
  item: {
    paddingVertical: 4,
  },
  itemRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
  },
  itemIcon: {
    marginRight: 12,
  },
  itemText: {
    flex: 1,
  },
  itemPrimary: {
    fontSize: 15,
    fontWeight: "500",
    color: "#333",
  },
  itemSecondary: {
    fontSize: 13,
    color: colors.textSecondary,
    marginTop: 2,
  },
  divider: {
    backgroundColor: colors.border,
  },
  closeButton: {
    marginTop: 16,
    borderRadius: 12,
  },
  closeButtonLabel: {
    fontSize: 16,
    fontWeight: "bold",
    paddingVertical: 4,
  },
});
