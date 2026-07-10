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
import CheckCircle from "@material-ui/icons/CheckCircle";
import ErrorIcon from "@material-ui/icons/Error";
import InfoIcon from "@material-ui/icons/Info";
import BarChart from "@material-ui/icons/BarChart";
import DISEASE_INFO from "../constants/diseaseInfo";
import { useStyles } from "../constants/theme";

const CLASS_COLORS = {
  "Early Blight": { primary: "#C62828", bg: "#FFEBEE", light: "#EF5350" },
  "Late Blight": { primary: "#E65100", bg: "#FFF3E0", light: "#FF9800" },
  "Healthy": { primary: "#2E7D32", bg: "#E8F5E9", light: "#66BB6A" },
};

function ProbabilityBarChart({ probabilities, predictedClass }) {
  if (!probabilities) return null;
  const entries = Object.entries(probabilities);

  return (
    <div style={{ marginTop: 24, width: '100%' }}>
      <Typography variant="subtitle1" style={{
        color: '#1B5E20', fontWeight: 700, marginBottom: 16,
        display: 'flex', alignItems: 'center', gap: 8, fontSize: 16,
      }}>
        <BarChart style={{ fontSize: 20 }} />
        Per-Class Probabilities
      </Typography>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {entries.map(([className, prob]) => {
          const pct = (prob * 100).toFixed(1);
          const isPredicted = className === predictedClass;
          const color = CLASS_COLORS[className] || { primary: '#666', bg: '#F5F5F5', light: '#999' };
          return (
            <div key={className} style={{
              padding: '10px 14px',
              backgroundColor: isPredicted ? color.bg : '#FAFAFA',
              borderRadius: 12,
              border: isPredicted ? `2px solid ${color.primary}` : '1px solid #E8E8E8',
              transition: 'all 0.3s ease',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <Typography variant="body2" style={{
                  fontWeight: isPredicted ? 700 : 500,
                  color: isPredicted ? color.primary : '#555',
                  fontSize: 14,
                }}>
                  {className} {isPredicted ? '←' : ''}
                </Typography>
                <Typography variant="body2" style={{
                  fontWeight: isPredicted ? 700 : 500,
                  color: isPredicted ? color.primary : '#888',
                  fontSize: 14,
                }}>
                  {pct}%
                </Typography>
              </div>
              <div style={{
                width: '100%', height: 8,
                backgroundColor: '#E8E8E8', borderRadius: 4, overflow: 'hidden',
              }}>
                <div style={{
                  width: `${pct}%`, height: '100%',
                  backgroundColor: color.light,
                  borderRadius: 4,
                  opacity: isPredicted ? 1 : 0.5,
                  transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                  boxShadow: isPredicted ? `0 0 8px ${color.light}` : 'none',
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
      <div style={{
        backgroundColor: '#F5FBF5',
        borderRadius: 16,
        padding: '20px 24px',
        border: '1px solid #C8E6C9',
      }}>
        <Typography variant="h6" style={{
          color: '#1B5E20', fontWeight: 700, marginBottom: 12,
          display: 'flex', alignItems: 'center', gap: 8, fontSize: 17,
        }}>
          <InfoIcon style={{ fontSize: 20, color: '#2E7D32' }} />
          About {diseaseClass}
        </Typography>
        <Typography variant="body2" style={{ marginBottom: 16, lineHeight: 1.7, color: '#444', fontSize: 14 }}>
          {info.description}
        </Typography>

        <Typography variant="subtitle1" style={{
          color: '#2E7D32', fontWeight: 600, marginBottom: 8, fontSize: 15,
        }}>
          {diseaseClass === "Healthy" ? "✅ Characteristics" : "🔍 Symptoms"}
        </Typography>
        <List dense style={{ padding: 0 }}>
          {info.symptoms.map((symptom, idx) => (
            <ListItem key={idx} style={{ padding: '3px 0' }}>
              <ListItemText primary={`• ${symptom}`} primaryTypographyProps={{ style: { color: '#444', fontSize: 13 } }} />
            </ListItem>
          ))}
        </List>

        <Typography variant="subtitle1" style={{
          color: '#2E7D32', fontWeight: 600, marginBottom: 8, marginTop: 16, fontSize: 15,
        }}>
          {diseaseClass === "Healthy" ? "💚 Care Tips" : "💊 Treatment & Mitigation"}
        </Typography>
        <List dense style={{ padding: 0 }}>
          {info.treatment.map((tip, idx) => (
            <ListItem key={idx} style={{ padding: '3px 0' }}>
              <ListItemText primary={`• ${tip}`} primaryTypographyProps={{ style: { color: '#444', fontSize: 13 } }} />
            </ListItem>
          ))}
        </List>
      </div>
    </div>
  );
}

function HeatmapSection({ heatmaps, heatmapLoading, selectedModel, modelNames }) {
  const classes = useStyles();
  const [showHeatmap, setShowHeatmap] = useState(false);
  const hasEnsemble = selectedModel === "ensemble";

  if (heatmapLoading) {
    return (
      <div className={classes.heatmapSection}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, justifyContent: 'center', padding: 20 }}>
          <CircularProgress size={20} className={classes.loader} />
          <Typography variant="body2" style={{ color: '#2E7D32', fontWeight: 600 }}>
            Computing heatmap...
          </Typography>
        </div>
      </div>
    );
  }

  if (!heatmaps) return null;

  return (
    <div className={classes.heatmapSection}>
      <div
        className={classes.heatmapHeader}
        onClick={() => setShowHeatmap(!showHeatmap)}
        style={{ cursor: 'pointer' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Whatshot style={{ color: '#E65100', fontSize: 22 }} />
          <Typography variant="subtitle1" style={{ color: '#1B5E20', fontWeight: 700, fontSize: 15 }}>
            Explainable AI — Heatmap
          </Typography>
        </div>
        {showHeatmap ? <ExpandLess style={{ color: '#666' }} /> : <ExpandMore style={{ color: '#666' }} />}
      </div>

      <Collapse in={showHeatmap}>
        <div className={classes.heatmapContent}>
          <Typography variant="body2" style={{
            color: '#666', marginBottom: 16, lineHeight: 1.6, fontSize: 13,
            padding: '12px 16px', backgroundColor: '#F5F5F5', borderRadius: 10,
          }}>
            🔥 The heatmap shows which parts of the image influenced the model's decision.
            <strong> Red areas</strong> = high importance, <strong>blue areas</strong> = low importance.
          </Typography>

          {hasEnsemble && heatmaps.heatmaps ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              {Object.entries(heatmaps.heatmaps).map(([modelName, overlayUrl]) => (
                <div key={modelName} style={{
                  backgroundColor: '#FFFFFF',
                  borderRadius: 14,
                  padding: 16,
                  border: '1px solid #E8F5E9',
                }}>
                  <Typography variant="caption" style={{
                    color: '#1B5E20', fontWeight: 700, marginBottom: 10,
                    display: 'block', fontSize: 12, textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}>
                    {modelName}
                  </Typography>
                  {overlayUrl ? (
                    <img
                      src={overlayUrl}
                      alt={`${modelName} heatmap`}
                      style={{
                        width: '100%', borderRadius: 12,
                        border: '2px solid #E8F5E9',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                      }}
                    />
                  ) : (
                    <div style={{
                      padding: 24, textAlign: 'center', backgroundColor: '#FAFAFA',
                      borderRadius: 10, color: '#999', fontSize: 13,
                    }}>
                      Heatmap unavailable for this model
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            heatmaps.heatmap && (
              <div style={{
                backgroundColor: '#FFFFFF', borderRadius: 14, padding: 12,
                border: '1px solid #E8F5E9',
              }}>
                <img
                  src={heatmaps.heatmap}
                  alt="Grad-CAM heatmap overlay"
                  style={{
                    width: '100%', borderRadius: 10,
                    border: '2px solid #E8F5E9',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                  }}
                />
              </div>
            )
          )}
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
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        gap: 12, marginBottom: 20,
      }}>
        <Warning style={{ color: '#F57C00', fontSize: 56 }} />
        <Typography variant="h5" style={{ color: '#E65100', fontWeight: 800, fontSize: 24 }}>
          Unknown Image
        </Typography>
      </div>
      <Typography variant="body1" style={{
        color: '#555', textAlign: 'center', marginBottom: 16, fontSize: 15, lineHeight: 1.6,
      }}>
        {data.message || "This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf."}
      </Typography>
      <div style={{
        display: 'inline-block', padding: '8px 20px',
        backgroundColor: '#FFF3E0', borderRadius: 20,
        color: '#E65100', fontWeight: 600, fontSize: 14,
        marginBottom: 20,
      }}>
        Max confidence: {confidence}%
      </div>
      <div style={{
        padding: 20, backgroundColor: '#FFF8E1',
        borderRadius: 14, border: '1px solid #FFE082',
        textAlign: 'left',
      }}>
        <Typography variant="subtitle2" style={{
          color: '#E65100', fontWeight: 700, marginBottom: 12, fontSize: 14,
        }}>
          Suggestions:
        </Typography>
        <List dense style={{ padding: 0 }}>
          {[
            'Upload a clear photo of a potato leaf',
            'Ensure the leaf occupies most of the frame',
            'Use good lighting for better results',
            'Avoid non-plant or blurred images',
          ].map((s, i) => (
            <ListItem key={i} style={{ padding: '3px 0' }}>
              <ListItemText primary={`• ${s}`} primaryTypographyProps={{ style: { color: '#795548', fontSize: 13 } }} />
            </ListItem>
          ))}
        </List>
      </div>
    </div>
  );
}

export default function PredictionResult({ data, heatmaps, heatmapLoading, selectedModel, modelNames }) {
  const classes = useStyles();

  if (data.is_unknown || data.class === "Unknown") {
    return (
      <CardContent className={classes.detail}>
        <UnknownImageCard data={data} />
      </CardContent>
    );
  }

  const confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  const isLowConfidence = parseFloat(confidence) < 70;
  const diseaseClass = data.class;
  const colors = CLASS_COLORS[diseaseClass] || CLASS_COLORS["Healthy"];

  return (
    <CardContent className={classes.detail}>
      {/* Prediction Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        gap: 12, marginBottom: 8, width: '100%',
      }}>
        {diseaseClass === "Healthy" ? (
          <CheckCircle style={{ color: colors.primary, fontSize: 32 }} />
        ) : (
          <ErrorIcon style={{ color: colors.primary, fontSize: 32 }} />
        )}
        <Typography variant="h5" style={{
          color: colors.primary, fontWeight: 800, fontSize: 22,
          letterSpacing: '-0.3px',
        }}>
          {diseaseClass}
        </Typography>
      </div>

      <Chip
        label={`${confidence}% Confidence`}
        style={{
          backgroundColor: colors.bg,
          color: colors.primary,
          fontWeight: 700,
          fontSize: 14,
          padding: '4px 8px',
          marginBottom: 20,
        }}
      />

      {/* Prediction Table */}
      <div style={{
        width: '100%',
        backgroundColor: colors.bg,
        borderRadius: 14,
        padding: '16px 20px',
        marginBottom: 4,
        border: `1px solid ${colors.bg}`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body1" style={{ fontWeight: 600, color: '#333', fontSize: 15 }}>
            Prediction
          </Typography>
          <Typography variant="body1" style={{ fontWeight: 700, color: colors.primary, fontSize: 15 }}>
            {confidence}%
          </Typography>
        </div>
      </div>

      {/* Per-Class Probabilities */}
      {data.probabilities && (
        <ProbabilityBarChart probabilities={data.probabilities} predictedClass={data.class} />
      )}

      {/* Low Confidence Warning */}
      {isLowConfidence && (
        <div style={{
          marginTop: 24, padding: '16px 20px',
          backgroundColor: '#FFF8E1', borderRadius: 14,
          border: '1px solid #FFE082',
          width: '100%',
        }}>
          <Typography variant="subtitle2" style={{
            color: '#E65100', fontWeight: 700, marginBottom: 10, fontSize: 14,
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <Warning style={{ fontSize: 20 }} />
            Low Confidence — Retake?
          </Typography>
          <Typography variant="body2" style={{ color: '#795548', fontSize: 13, lineHeight: 1.6 }}>
            The model isn't very confident. Try retaking the photo with better lighting
            and ensure the leaf is clear and centered.
          </Typography>
        </div>
      )}

      {/* Individual Model Predictions */}
      {data.individual && (
        <div style={{ marginTop: 24, width: '100%' }}>
          <Typography variant="subtitle1" style={{
            color: '#1B5E20', fontWeight: 700, marginBottom: 14, fontSize: 16,
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <InfoIcon style={{ fontSize: 20 }} />
            Individual Model Results
          </Typography>
          <TableContainer component={Paper} className={classes.tableContainer}>
            <Table size="small">
              <TableHead className={classes.tableHead}>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>Prediction</TableCell>
                  <TableCell align="right">Confidence</TableCell>
                </TableRow>
              </TableHead>
              <TableBody className={classes.tableBody}>
                {data.individual.map((pred, idx) => {
                  const c = CLASS_COLORS[pred.class] || CLASS_COLORS["Healthy"];
                  return (
                    <TableRow key={idx}>
                      <TableCell style={{ fontWeight: 600, color: '#333', fontSize: 13 }}>
                        {pred.model}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={pred.class}
                          size="small"
                          style={{
                            backgroundColor: c.bg,
                            color: c.primary,
                            fontWeight: 600,
                            fontSize: 12,
                          }}
                        />
                      </TableCell>
                      <TableCell align="right" style={{ fontWeight: 600, color: '#333', fontSize: 13 }}>
                        {(pred.confidence * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
      )}

      {/* Grad-CAM */}
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
