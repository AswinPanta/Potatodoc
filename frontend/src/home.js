
import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper, CardActionArea, CardMedia, Grid, TableContainer, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress, Select, MenuItem, FormControl, InputLabel, Dialog, DialogTitle, DialogContent, List, ListItem, ListItemText, IconButton } from "@material-ui/core";
import image from "./bg.png";
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import Clear from '@material-ui/icons/Clear';
import History from '@material-ui/icons/History';
import CameraAlt from '@material-ui/icons/CameraAlt';

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText("#8B4513"),
    backgroundColor: "#8B4513",
    '&:hover': {
      backgroundColor: "#A0522D",
    },
    borderRadius: 12,
    padding: "12px 24px",
    textTransform: "none",
    fontWeight: "bold",
  },
}))(Button);

const axios = require("axios").default;

// Disease Information
const DISEASE_INFO = {
  "Early Blight": {
    description: "Early blight is a common fungal disease caused by Alternaria solani. It affects leaves, stems, and fruits of potato plants.",
    symptoms: [
      "Small, brown to black lesions with concentric rings on older leaves",
      "Yellowing of leaf tissue around the lesions",
      "Lesions may coalesce, causing large areas of leaf tissue to die",
      "Defoliation in severe cases"
    ],
    treatment: [
      "Remove and destroy infected plant debris",
      "Use crop rotation with non-host crops",
      "Apply fungicides (like mancozeb or chlorothalonil) according to label instructions",
      "Ensure good air circulation by proper spacing of plants",
      "Avoid overhead watering"
    ]
  },
  "Late Blight": {
    description: "Late blight is a devastating fungal disease caused by Phytophthora infestans. It can destroy entire potato fields in a short time.",
    symptoms: [
      "Dark green to brown, water-soaked lesions on leaves",
      "White, fuzzy growth on the undersides of leaves in wet conditions",
      "Lesions expand rapidly and turn dark brown to black",
      "Brown lesions on stems and tubers"
    ],
    treatment: [
      "Remove and destroy all infected plants immediately",
      "Apply fungicides (like copper-based products or systemic fungicides) as a preventive measure",
      "Use resistant potato varieties",
      "Avoid overhead watering; use drip irrigation instead",
      "Monitor weather conditions; late blight thrives in cool, wet weather"
    ]
  },
  "Healthy": {
    description: "Your potato plant is healthy! Continue to practice good crop care to keep it that way.",
    symptoms: [
      "Vibrant green leaves with no lesions or discoloration",
      "Strong, upright stems",
      "No visible signs of pests or diseases"
    ],
    treatment: [
      "Maintain a regular watering schedule",
      "Fertilize appropriately for potato plants",
      "Monitor for any signs of pests or diseases",
      "Practice crop rotation",
      "Ensure good soil health"
    ]
  }
};

const useStyles = makeStyles((theme) => ({
  grow: {
    flexGrow: 1,
  },
  clearButton: {
    width: "-webkit-fill-available",
  },
  root: {
    maxWidth: 345,
    flexGrow: 1,
  },
  media: {
    height: 400,
    borderRadius: 12,
  },
  paper: {
    padding: theme.spacing(2),
    margin: 'auto',
    maxWidth: 500,
  },
  gridContainer: {
    justifyContent: "center",
    padding: "2em 1em 1em 1em",
  },
  mainContainer: {
    backgroundImage: `linear-gradient(to bottom right, #E8F5E9, #FFF8E1)`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    minHeight: "100vh",
    marginTop: 0,
    paddingTop: 24,
  },
  imageCard: {
    margin: "auto",
    maxWidth: 480,
    width: "95%",
    height: "auto",
    backgroundColor: 'white',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
    borderRadius: 20,
    overflow: "hidden",
  },
  imageCardEmpty: {
    height: 'auto',
  },
  noImage: {
    margin: "auto",
    width: 400,
    height: "400 !important",
  },
  input: {
    display: 'none',
  },
  uploadIcon: {
    background: 'white',
  },
  tableContainer: {
    backgroundColor: 'transparent !important',
    boxShadow: 'none !important',
  },
  table: {
    backgroundColor: 'transparent !important',
  },
  tableHead: {
    backgroundColor: '#E8F5E9 !important',
  },
  tableRow: {
    backgroundColor: 'transparent !important',
  },
  tableCell: {
    fontSize: '20px',
    backgroundColor: 'transparent !important',
    borderColor: 'transparent !important',
    color: '#2E7D32 !important',
    fontWeight: 'bolder',
    padding: '12px 24px',
  },
  tableCell1: {
    fontSize: '14px',
    backgroundColor: 'transparent !important',
    borderColor: '#C8E6C9 !important',
    color: '#1B5E20 !important',
    fontWeight: '600',
    padding: '10px 24px',
  },
  tableBody: {
    backgroundColor: 'transparent !important',
  },
  text: {
    color: '#1B5E20 !important',
    textAlign: 'center',
  },
  buttonGrid: {
    maxWidth: "480px",
    width: "100%",
    marginTop: 16,
  },
  detail: {
    backgroundColor: 'white',
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: 'center',
    padding: 20,
  },
  appbar: {
    background: 'linear-gradient(90deg, #2E7D32 0%, #388E3C 50%, #43A047 100%)',
    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)',
    color: 'white',
    padding: '4px 0',
  },
  loader: {
    color: '#2E7D32 !important',
  },
  formControl: {
    margin: theme.spacing(2, 0),
    minWidth: 220,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: '4px 12px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
    width: "100%",
  },
  logo: {
    width: 48,
    height: 48,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    fontFamily: "'Poppins', sans-serif",
  },
  dropzone: {
    border: '2px dashed #81C784',
    borderRadius: 16,
    backgroundColor: '#F1F8F2',
    padding: 24,
    minHeight: 200,
  },
  uploadOptions: {
    display: 'flex',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 16,
  },
  infoSection: {
    marginTop: 24,
    width: '100%',
  },
  infoTitle: {
    color: '#2E7D32',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  infoList: {
    paddingLeft: 20,
  },
  infoListItem: {
    padding: '4px 0',
  },
  historyButton: {
    color: 'white',
  },
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  const [models, setModels] = useState(["cnn-baseline", "transfer-learning", "mobilenetv2", "ensemble"]);
  const [modelNames, setModelNames] = useState({
    "cnn-baseline": "CNN Baseline",
    "transfer-learning": "Transfer Learning",
    "mobilenetv2": "MobileNetV2",
  });
  const [selectedModel, setSelectedModel] = useState("ensemble");
  const [historyDialog, setHistoryDialog] = useState(false);
  const [history, setHistory] = useState([]);
  const [showWebcam, setShowWebcam] = useState(false);
  const [webcamStream, setWebcamStream] = useState(null);
  const webcamRef = React.useRef(null);
  const canvasRef = React.useRef(null);

  // Load history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('potatoDocHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Fetch available models from API on load
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await axios.get("http://localhost:8000/models");
        setModels([...res.data.models, "ensemble"]);
        setModelNames(res.data.modelNames);
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    };
    fetchModels();
  }, []);

  // Cleanup webcam stream
  useEffect(() => {
    return () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [webcamStream]);

  // Save to history
  const saveToHistory = (newPrediction) => {
    const newHistory = [
      {
        id: Date.now(),
        ...newPrediction,
        timestamp: new Date().toLocaleString()
      },
      ...history
    ].slice(0, 20); // Keep last 20
    setHistory(newHistory);
    localStorage.setItem('potatoDocHistory', JSON.stringify(newHistory));
  };

  const sendFile = async () => {
    if (image) {
      let formData = new FormData();
      formData.append("file", selectedFile);
      try {
        let res = await axios({
          method: "post",
          url: `http://localhost:8000/predict?model_id=${selectedModel}`,
          data: formData,
        });
        if (res.status === 200) {
          setData(res.data);
          saveToHistory({
            ...res.data,
            model: selectedModel === "ensemble" ? "Ensemble" : modelNames[selectedModel]
          });
        }
      } catch (err) {
        console.error("Prediction failed:", err);
      }
      setIsloading(false);
    }
  };

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
    setShowWebcam(false);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    setIsloading(true);
    sendFile();
  }, [preview, selectedModel]); // Re-run when model changes

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setImage(true);
    setShowWebcam(false);
  };

  // Webcam functions
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setWebcamStream(stream);
      setShowWebcam(true);
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };

  const captureImage = () => {
    if (webcamRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = webcamRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      canvas.toBlob((blob) => {
        const file = new File([blob], `webcam-${Date.now()}.png`, { type: 'image/png' });
        setSelectedFile(file);
        setData(undefined);
        setImage(true);
        setShowWebcam(false);
      }, 'image/png');
    }
  };

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
      setShowWebcam(false);
    }
  };

  let confidence = 0;
  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar} elevation={0}>
        <Toolbar>
          <Avatar src="/icon.png" className={classes.logo} />
          <Typography className={`${classes.grow} ${classes.title}`} variant="h5" noWrap>
            PotatoDoc
          </Typography>
          <IconButton className={classes.historyButton} onClick={() => setHistoryDialog(true)}>
            <History />
          </IconButton>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" className={classes.mainContainer} disableGutters={true}>
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="flex-start"
          spacing={3}
        >
          <Grid item xs={12} md={8} lg={6}>
            <Card className={`${classes.imageCard} ${!image ? classes.imageCardEmpty : ''}`}>
              {/* Model Selection Dropdown */}
              <CardContent style={{ paddingBottom: 0 }}>
                <FormControl variant="outlined" className={classes.formControl}>
                  <InputLabel id="model-select-label" style={{ color: '#2E7D32', fontWeight: 600 }}>Select Model</InputLabel>
                  <Select
                    labelId="model-select-label"
                    id="model-select"
                    value={selectedModel}
                    label="Select Model"
                    onChange={(e) => {
                      setSelectedModel(e.target.value);
                      if (data) {
                        setData(null); // Re-predict if we have data
                      }
                    }}
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

              {showWebcam && (
                <CardContent>
                  <div style={{ position: 'relative', textAlign: 'center' }}>
                    <video ref={webcamRef} autoPlay playsInline style={{ width: '100%', borderRadius: 12 }} />
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                    <div style={{ marginTop: 16, display: 'flex', gap: 12, justifyContent: 'center' }}>
                      <ColorButton variant="contained" onClick={captureImage} startIcon={<CameraAlt />}>
                        Capture
                      </ColorButton>
                      <ColorButton variant="contained" onClick={stopWebcam} startIcon={<Clear />}>
                        Cancel
                      </ColorButton>
                    </div>
                  </div>
                </CardContent>
              )}

              {image && !showWebcam && <CardActionArea style={{ padding: '0 16px 16px' }}>
                <CardMedia
                  className={classes.media}
                  image={preview}
                  component="img"
                  title="Potato Leaf"
                />
              </CardActionArea>}

              {!image && !showWebcam && <CardContent className={classes.content} style={{ padding: '16px 24px 24px' }}>
                <div className={classes.uploadOptions}>
                  <ColorButton variant="contained" onClick={startWebcam} startIcon={<CameraAlt />}>
                    Take Photo
                  </ColorButton>
                </div>
                <DropzoneArea
                  acceptedFiles={['image/*']}
                  dropzoneText={"Drag & drop a potato leaf image here or click to upload"}
                  onChange={onSelectFile}
                  dropzoneClass={classes.dropzone}
                  filesLimit={1}
                  showFileNames={true}
                  showAlerts={false}
                />
                <Typography variant="body2" style={{ marginTop: 12, color: '#666', textAlign: 'center' }}>
                  Supported formats: JPEG, PNG, GIF • Max file size: 5MB
                </Typography>

                {/* Image Capture Tips */}
                <div style={{ marginTop: 20, padding: 16, backgroundColor: '#E8F5E9', borderRadius: 12 }}>
                  <Typography variant="subtitle2" style={{ color: '#2E7D32', fontWeight: 'bold', marginBottom: 8 }}>
                    Tips for Taking Good Photos:
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="• Capture the entire potato leaf in focus" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="• Use good, natural lighting (avoid harsh shadows)" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="• Place the leaf on a plain, neutral background" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="• Make sure the leaf is the main subject of the photo" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="• Capture both healthy and diseased parts if present" />
                    </ListItem>
                  </List>
                </div>
              </CardContent>}

              {data && <CardContent className={classes.detail}>
                {/* Main Prediction */}
                <Typography variant="h5" gutterBottom style={{ color: '#1B5E20', fontWeight: 'bold', marginBottom: 16 }}>
                  Prediction Result
                </Typography>
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
                        <TableCell component="th" scope="row" className={classes.tableCell}>
                          {data.class}
                        </TableCell>
                        <TableCell align="right" className={classes.tableCell}>{confidence}%</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* Low Confidence Warning */}
                {parseFloat(confidence) < 70 && (
                  <div style={{ marginTop: 20, padding: 16, backgroundColor: '#FFF3CD', borderRadius: 12, border: '1px solid #FFEEBA' }}>
                    <Typography variant="subtitle2" style={{ color: '#856404', fontWeight: 'bold', marginBottom: 8 }}>
                      ⚠️ Low Confidence Prediction
                    </Typography>
                    <Typography variant="body2" style={{ color: '#856404' }}>
                      The model is not very confident in this prediction. Please try retaking the photo using the tips below:
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText primary="• Make sure the photo shows a clear potato leaf" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="• Use better natural lighting" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="• Ensure the leaf is in focus and centered" />
                      </ListItem>
                      <ListItem>
                        <ListItemText primary="• Avoid blurry or dark photos" />
                      </ListItem>
                    </List>
                  </div>
                )}

                {/* Individual Model Predictions for Ensemble */}
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

                {/* Disease Info */}
                {DISEASE_INFO[data.class] && (
                  <div className={classes.infoSection}>
                    <Typography variant="h6" gutterBottom className={classes.infoTitle}>
                      About {data.class}
                    </Typography>
                    <Typography variant="body2" style={{ marginBottom: 16 }}>
                      {DISEASE_INFO[data.class].description}
                    </Typography>
                    <Typography variant="subtitle1" className={classes.infoTitle} style={{ marginBottom: 8 }}>
                      Symptoms:
                    </Typography>
                    <List className={classes.infoList} dense>
                      {DISEASE_INFO[data.class].symptoms.map((symptom, idx) => (
                        <ListItem key={idx} className={classes.infoListItem}>
                          <ListItemText primary={`• ${symptom}`} />
                        </ListItem>
                      ))}
                    </List>
                    <Typography variant="subtitle1" className={classes.infoTitle} style={{ marginBottom: 8, marginTop: 16 }}>
                      {data.class === "Healthy" ? "Care Tips:" : "Treatment & Mitigation:"}
                    </Typography>
                    <List className={classes.infoList} dense>
                      {DISEASE_INFO[data.class].treatment.map((tip, idx) => (
                        <ListItem key={idx} className={classes.infoListItem}>
                          <ListItemText primary={`• ${tip}`} />
                        </ListItem>
                      ))}
                    </List>
                  </div>
                )}
              </CardContent>}

              {isLoading && <CardContent className={classes.detail}>
                <CircularProgress color="secondary" className={classes.loader} size={48} />
                <Typography className={classes.title} variant="h6" noWrap style={{ marginTop: 16, color: '#1B5E20' }}>
                  Processing...
                </Typography>
              </CardContent>}
            </Card>
          </Grid>

          {data && <Grid item xs={12} md={8} lg={6} className={classes.buttonGrid}>
            <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="large" onClick={clearData} startIcon={<Clear fontSize="large" />}>
              Clear
            </ColorButton>
          </Grid>}
        </Grid>
      </Container>

      {/* History Dialog */}
      <Dialog open={historyDialog} onClose={() => setHistoryDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Prediction History</DialogTitle>
        <DialogContent>
          <List>
            {history.length === 0 ? (
              <ListItem>
                <ListItemText primary="No predictions yet" secondary="Start predicting to see history here" />
              </ListItem>
            ) : (
              history.map((item) => (
                <ListItem key={item.id} divider>
                  <ListItemText
                    primary={`${item.class} (${(item.confidence * 100).toFixed(2)}%) - ${item.model}`}
                    secondary={item.timestamp}
                  />
                </ListItem>
              ))
            )}
          </List>
        </DialogContent>
      </Dialog>
    </React.Fragment>
  );
};
