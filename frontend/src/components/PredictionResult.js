import React, { useState } from "react";
import CardContent from "@material-ui/core/CardContent";
import Typography from "@material-ui/core/Typography";
import Paper from "@material-ui/core/Paper";
import TableContainer from "@material-ui/core/TableContainer";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import Collapse from "@material-ui/core/Collapse";
import Chip from "@material-ui/core/Chip";
import CircularProgress from "@material-ui/core/CircularProgress";
import ExpandMore from "@material-ui/icons/ExpandMore";
import ExpandLess from "@material-ui/icons/ExpandLess";
import Whatshot from "@material-ui/icons/Whatshot";
import Warning from "@material-ui/icons/Warning";
import DISEASE_INFO from "../constants/diseaseInfo";
import { useStyles } from "../constants/theme";

const CLASS_COLORS = {
  "Early Blight": "#C62828",
  "Late Blight": "#E65100",
  "Healthy": "#2E7D32",
};

function ProbabilityBarChart({ probabilities, predictedClass }) {
  const classes = useStyles();

  if (!probabilities) return null;

  const entries = Object.entries(probabilities);

  return (
    <div style={{ marginTop: 20, width: '100%' }}>
      <Typography variant="subtitle1" style={{ color: '#1B5E20', fontWeight: 'bold', marginBottom: 12 }}>
        Per-Class Probabilities
      </Typography>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {entries.map(([className, prob]) => {
          const pct = (prob * 100).toFixed(1);
          const isPredicted = className === predictedClass;
          const barColor = CLASS_COLORS[className] || '#666';
          return (
            <div key={className}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <Typography variant="body2" style={{
                  fontWeight: isPredicted ? 'bold' : 'normal',
                  color: isPredicted ? barColor : '#555',
                  fontSize: 13,
                }}>
                  {className} {isPredicted ? '←' : ''}
                </Typography>
                <Typography variant="body2" style={{
                  fontWeight: isPredicted ? 'bold' : 'normal',
                  color: isPredicted ? barColor : '#555',
                  fontSize: 13,
                }}>
                  {pct}%
                </Typography>
              </div>
              <div style={{
                width: '100%',
                height: 10,
                backgroundColor: '#E8E8E8',
                borderRadius: 5,
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${pct}%`,
                  height: '100%',
                  backgroundColor: barColor,
                  borderRadius: 5,
                  opacity: isPredicted ? 1 : 0.4,
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DiseaseInfoCard({ diseaseClass }) {
  const classes = useStyles();
  const info = DISEASE_INFO[diseaseClass];
  if (!info) return null;

  return (
    <div className={classes.infoSection}>
      <Typography variant="h6" gutterBottom className={classes.infoTitle}>
        About {diseaseClass}
      </Typography>
      <Typography variant="body2" style={{ marginBottom: 16, lineHeight: 1.6, color: '#444' }}>
        {info.description}
      </Typography>
      <Typography variant="subtitle1" className={classes.infoTitle} style={{ marginBottom: 8 }}>
        Symptoms:
      </Typography>
      <List className={classes.infoList} dense>
        {info.symptoms.map((symptom, idx) => (
          <ListItem key={idx} className={classes.infoListItem}>
            <ListItemText primary={`• ${symptom}`} />
          </ListItem>
        ))}
      </List>
      <Typography variant="subtitle1" className={classes.infoTitle} style={{ marginBottom: 8, marginTop: 16 }}>
        {diseaseClass === "Healthy" ? "Care Tips:" : "Treatment & Mitigation:"}
      </Typography>
      <List className={classes.infoList} dense>
        {info.treatment.map((tip, idx) => (
          <ListItem key={idx} className={classes.infoListItem}>
            <ListItemText primary={`• ${tip}`} />
          </ListItem>
        ))}
      </List>
    </div>
  );
}

function HeatmapSection({ heatmaps, heatmapLoading, selectedModel, modelNames }) {
  const classes = useStyles();
  const [showHeatmap, setShowHeatmap] = useState(false);

  if (heatmapLoading) {
    return (
      <div className={classes.heatmapSection}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center', padding: 12 }}>
          <CircularProgress size={18} className={classes.loader} />
          <Typography variant="body2" style={{ color: '#2E7D32' }}>
            Computing heatmap...
          </Typography>
        </div>
      </div>
    );
  }

  if (!heatmaps) return null;

  const hasEnsemble = selectedModel === "ensemble";

  return (
    <div className={classes.heatmapSection}>
      <div
        className={classes.heatmapHeader}
        onClick={() => setShowHeatmap(!showHeatmap)}
        style={{ cursor: 'pointer' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Whatshot style={{ color: '#E65100' }} />
          <Typography variant="subtitle1" style={{ color: '#2E7D32', fontWeight: 'bold' }}>
            Explainable AI - Heatmap
          </Typography>
        </div>
        {showHeatmap ? <ExpandLess /> : <ExpandMore />}
      </div>

      <Collapse in={showHeatmap}>
        <div className={classes.heatmapContent}>
          <Typography variant="body2" style={{ color: '#666', marginBottom: 12, lineHeight: 1.5 }}>
            The heatmap highlights which regions of the image the model focused on
            to make its prediction. Red areas indicate high importance, blue areas low importance.
          </Typography>

          {hasEnsemble && heatmaps.heatmaps ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {Object.entries(heatmaps.heatmaps).map(([modelName, overlayUrl]) => (
                <div key={modelName}>
                  <Typography variant="caption" style={{ color: '#555', fontWeight: 600, marginBottom: 8, display: 'block' }}>
                    {modelName}
                  </Typography>
                  {overlayUrl ? (
                    <img
                      src={overlayUrl}
                      alt={`${modelName} Grad-CAM heatmap`}
                      style={{ width: '100%', borderRadius: 12, border: '2px solid #E8F5E9' }}
                    />
                  ) : (
                    <Typography variant="caption" style={{ color: '#999' }}>
                      Heatmap unavailable for this model
                    </Typography>
                  )}
                </div>
              ))}
            </div>
          ) : (
            heatmaps.heatmap && (
              <img
                src={heatmaps.heatmap}
                alt="Grad-CAM heatmap overlay"
                style={{ width: '100%', borderRadius: 12, border: '2px solid #E8F5E9' }}
              />
            )
          )}

          <Typography variant="caption" style={{ color: '#999', marginTop: 8, display: 'block', textAlign: 'center' }}>
            Grad-CAM visualization • Red = high importance
          </Typography>
        </div>
      </Collapse>
    </div>
  );
}

function UnknownImageCard({ data }) {
  const classes = useStyles();
  const confidence = (parseFloat(data.confidence) * 100).toFixed(2);

  return (
    <div className={classes.unknownCard}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, marginBottom: 12 }}>
        <Warning style={{ color: '#F57C00', fontSize: 40 }} />
        <Typography variant="h5" style={{ color: '#E65100', fontWeight: 'bold' }}>
          Unknown Image
        </Typography>
      </div>
      <Typography variant="body1" style={{ color: '#555', textAlign: 'center', marginBottom: 12 }}>
        {data.message || "This does not appear to be a potato leaf."}
      </Typography>
      <Typography variant="body2" style={{ color: '#888', textAlign: 'center' }}>
        Max confidence: {confidence}%
      </Typography>
      <div style={{ marginTop: 16, padding: 16, backgroundColor: '#FFF3E0', borderRadius: 12 }}>
        <Typography variant="subtitle2" style={{ color: '#E65100', fontWeight: 'bold', marginBottom: 8 }}>
          Suggestions:
        </Typography>
        <List dense>
          <ListItem><ListItemText primary="• Upload a clear photo of a potato leaf" /></ListItem>
          <ListItem><ListItemText primary="• Ensure the leaf occupies most of the frame" /></ListItem>
          <ListItem><ListItemText primary="• Use good lighting for better results" /></ListItem>
          <ListItem><ListItemText primary="• Avoid non-plant or blurred images" /></ListItem>
        </List>
      </div>
    </div>
  );
}

export default function PredictionResult({ data, heatmaps, heatmapLoading, selectedModel, modelNames }) {
  const classes = useStyles();

  // Handle unknown image
  if (data.is_unknown || data.class === "Unknown") {
    return (
      <CardContent className={classes.detail}>
        <UnknownImageCard data={data} />
      </CardContent>
    );
  }

  const confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  const isLowConfidence = parseFloat(confidence) < 70;

  // Determine color based on class
  const resultColor = data.class === "Healthy" ? '#2E7D32' : '#C62828';

  return (
    <CardContent className={classes.detail}>
      <Typography variant="h5" gutterBottom style={{ color: resultColor, fontWeight: 'bold', marginBottom: 16 }}>
        Prediction Result
      </Typography>

      {/* Main Prediction Table */}
      <TableContainer component={Paper} className={classes.tableContainer} style={{ borderRadius: 12, overflow: 'hidden', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)' }}>
        <Table className={classes.table} size="small" aria-label="simple table">
          <TableHead className={classes.tableHead}>
            <TableRow className={classes.tableRow}>
              <TableCell className={classes.tableCell1}>Label</TableCell>
              <TableCell align="right" className={classes.tableCell1}>Confidence</TableCell>
            </TableRow>
          </TableHead>
          <TableBody className={classes.tableBody}>
            <TableRow className={classes.tableRow}>
              <TableCell component="th" scope="row" className={classes.tableCell} style={{ color: resultColor }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Chip
                    label={data.class}
                    style={{
                      backgroundColor: data.class === "Healthy" ? '#E8F5E9' : '#FFEBEE',
                      color: resultColor,
                      fontWeight: 'bold',
                      fontSize: 14,
                    }}
                  />
                </div>
              </TableCell>
              <TableCell align="right" className={classes.tableCell} style={{ color: resultColor }}>
                {confidence}%
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>

      {/* Per-Class Probabilities Bar Chart */}
      {data.probabilities && (
        <ProbabilityBarChart
          probabilities={data.probabilities}
          predictedClass={data.class}
        />
      )}

      {/* Low Confidence Warning */}
      {isLowConfidence && (
        <div style={{ marginTop: 20, padding: 16, backgroundColor: '#FFF3CD', borderRadius: 12, border: '1px solid #FFEEBA' }}>
          <Typography variant="subtitle2" style={{ color: '#856404', fontWeight: 'bold', marginBottom: 8 }}>
            ⚠️ Low Confidence Prediction
          </Typography>
          <Typography variant="body2" style={{ color: '#856404' }}>
            The model is not very confident in this prediction. Please try retaking the photo:
          </Typography>
          <List dense>
            <ListItem><ListItemText primary="• Make sure the photo shows a clear potato leaf" /></ListItem>
            <ListItem><ListItemText primary="• Use better natural lighting" /></ListItem>
            <ListItem><ListItemText primary="• Ensure the leaf is in focus and centered" /></ListItem>
            <ListItem><ListItemText primary="• Avoid blurry or dark photos" /></ListItem>
          </List>
        </div>
      )}

      {/* Individual Model Predictions (Ensemble) */}
      {data.individual && (
        <div style={{ marginTop: 24, width: '100%' }}>
          <Typography variant="h6" gutterBottom style={{ color: '#1B5E20', fontWeight: 'bold', marginBottom: 12 }}>
            Individual Model Predictions
          </Typography>
          <TableContainer component={Paper} className={classes.tableContainer} style={{ borderRadius: 12, overflow: 'hidden', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)' }}>
            <Table className={classes.table} size="small">
              <TableHead className={classes.tableHead}>
                <TableRow className={classes.tableRow}>
                  <TableCell className={classes.tableCell1}>Model</TableCell>
                  <TableCell className={classes.tableCell1}>Label</TableCell>
                  <TableCell align="right" className={classes.tableCell1}>Confidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody className={classes.tableBody}>
                {data.individual.map((pred, idx) => (
                  <TableRow key={idx} className={classes.tableRow}>
                    <TableCell className={classes.tableCell1}>{pred.model}</TableCell>
                    <TableCell className={classes.tableCell1}>{pred.class}</TableCell>
                    <TableCell align="right" className={classes.tableCell1}>{(pred.confidence * 100).toFixed(2)}%</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
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
    </CardContent>
  );
}
