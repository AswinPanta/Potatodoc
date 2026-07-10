import React from "react";
import CardContent from "@material-ui/core/CardContent";
import FormControl from "@material-ui/core/FormControl";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import { useStyles } from "../constants/theme";

export default function ModelSelector({ models, modelNames, selectedModel, onChange }) {
  const classes = useStyles();

  return (
    <CardContent style={{ paddingBottom: 0 }}>
      <FormControl variant="outlined" className={classes.formControl}>
        <InputLabel id="model-select-label" style={{ color: '#2E7D32', fontWeight: 600 }}>Select Model</InputLabel>
        <Select
          labelId="model-select-label"
          id="model-select"
          value={selectedModel}
          label="Select Model"
          onChange={(e) => onChange(e.target.value)}
          style={{ color: '#1B5E20', fontWeight: 600 }}
        >
          {models.map((model) => (
            <MenuItem key={model} value={model} style={{ fontWeight: 500 }}>
              {model === "ensemble" ? "Ensemble (All Models)" : modelNames[model]}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </CardContent>
  );
}
