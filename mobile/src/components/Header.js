import React from "react";
import { StyleSheet } from "react-native";
import { Appbar } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors } from "../constants/colors";

export default function Header({ onHistoryPress }) {
  return (
    <Appbar.Header style={styles.appbar}>
      <Appbar.Content
        title="PotatoDoc"
        titleStyle={styles.appbarTitle}
      />
      <Appbar.Action
        icon={() => (
          <MaterialCommunityIcons
            name="history"
            size={24}
            color="white"
          />
        )}
        onPress={onHistoryPress}
      />
    </Appbar.Header>
  );
}

const styles = StyleSheet.create({
  appbar: {
    backgroundColor: colors.primary,
    elevation: 4,
  },
  appbarTitle: {
    fontWeight: "bold",
    fontSize: 22,
  },
});
