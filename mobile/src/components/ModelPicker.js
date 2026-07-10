import React from "react";
import { View, StyleSheet } from "react-native";
import { Button, Menu, Text } from "react-native-paper";
import { colors } from "../constants/colors";

export default function ModelPicker({ models, modelNames, selectedModel, onChange }) {
  const [visible, setVisible] = React.useState(false);

  return (
    <View style={styles.container}>
      <Menu
        visible={visible}
        onDismiss={() => setVisible(false)}
        anchor={
          <Button
            mode="outlined"
            onPress={() => setVisible(true)}
            style={styles.button}
            contentStyle={styles.buttonContent}
            labelStyle={styles.buttonLabel}
            textColor={colors.textPrimary}
          >
            {selectedModel === "ensemble"
              ? "Ensemble (All Models)"
              : modelNames[selectedModel] || selectedModel}
          </Button>
        }
      >
        {models.map((model) => (
          <Menu.Item
            key={model}
            onPress={() => {
              onChange(model);
              setVisible(false);
            }}
            title={
              model === "ensemble"
                ? "Ensemble (All Models)"
                : modelNames[model] || model
            }
            titleStyle={{
              fontWeight: selectedModel === model ? "bold" : "normal",
              color: selectedModel === model ? colors.primary : undefined,
            }}
          />
        ))}
      </Menu>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    marginVertical: 8,
  },
  button: {
    borderColor: colors.border,
    borderRadius: 12,
    minWidth: 220,
    backgroundColor: colors.surface,
  },
  buttonContent: {
    paddingVertical: 4,
  },
  buttonLabel: {
    fontSize: 15,
    fontWeight: "600",
  },
});
