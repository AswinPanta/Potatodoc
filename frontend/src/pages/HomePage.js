import React, { useState, useEffect, useRef } from "react";
import Container from "@material-ui/core/Container";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import CardMedia from "@material-ui/core/CardMedia";
import CardActionArea from "@material-ui/core/CardActionArea";
import Grid from "@material-ui/core/Grid";
import CircularProgress from "@material-ui/core/CircularProgress";
import Typography from "@material-ui/core/Typography";
import PhotoCamera from "@material-ui/icons/PhotoCamera";
import SaveIcon from "@material-ui/icons/Save";
import Refresh from "@material-ui/icons/Refresh";
import Navbar from "../components/Navbar";
import ModelSelector from "../components/ModelSelector";
import ImageInput from "../components/ImageInput";
import WebcamCapture from "../components/WebcamCapture";
import PredictionResult from "../components/PredictionResult";
import HistoryDialog from "../components/HistoryDialog";
import { useStyles, ColorButton, SecondaryButton } from "../constants/theme";
import { useModels, usePrediction, fetchGradcam } from "../hooks/useApi";
import { useHistory } from "../hooks/useHistory";

export default function HomePage() {
  const classes = useStyles();
  const { models, modelNames } = useModels();
  const { data, setData, isLoading, sendFile } = usePrediction();
  const { history, saveToHistory } = useHistory();

  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [image, setImage] = useState(false);
  const [selectedModel, setSelectedModel] = useState("ensemble");
  const [historyDialog, setHistoryDialog] = useState(false);
  const [showWebcam, setShowWebcam] = useState(false);
  const [webcamStream, setWebcamStream] = useState(null);
  const [heatmaps, setHeatmaps] = useState(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const [fadeIn, setFadeIn] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    fileRef.current = selectedFile;
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [webcamStream]);

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) return;
    const doPrediction = async () => {
      setHeatmaps(null);
      setSaved(false);
      setFadeIn(false);
      await sendFile(selectedFile, selectedModel);
      setTimeout(() => setFadeIn(true), 100);
    };
    doPrediction();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preview, selectedModel]);

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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

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
    setFadeIn(false);
  };

  const nextImage = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
    setShowWebcam(false);
    setHeatmaps(null);
    setSaved(false);
    setFadeIn(false);
  };

  const handleSaveToHistory = () => {
    saveToHistory({
      ...data,
      model: selectedModel === "ensemble" ? "Ensemble" : modelNames[selectedModel],
      heatmap: heatmaps,
    });
    setSaved(true);
  };

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

  const handleModelChange = (value) => {
    setSelectedModel(value);
    if (data) {
      setData(null);
    }
    setHeatmaps(null);
    setSaved(false);
    setFadeIn(false);
  };

  const hasSaveableResult = data && !data.error && data.class !== "Unknown";

  return (
    <React.Fragment>
      <Navbar onHistoryClick={() => setHistoryDialog(true)} />
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
            <Card className={classes.imageCard}>
              <ModelSelector
                models={models}
                modelNames={modelNames}
                selectedModel={selectedModel}
                onChange={handleModelChange}
              />

              {showWebcam && (
                <WebcamCapture
                  webcamRef={webcamRef}
                  canvasRef={canvasRef}
                  onCapture={captureImage}
                  onCancel={stopWebcam}
                />
              )}

              {image && !showWebcam && (
                <CardActionArea style={{ padding: '16px 20px 8px' }}>
                  <div style={{
                    borderRadius: 16, overflow: 'hidden',
                    boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
                  }}>
                    <CardMedia
                      className={classes.media}
                      image={preview}
                      component="img"
                      title="Potato Leaf"
                      style={{ maxHeight: 380 }}
                    />
                  </div>
                </CardActionArea>
              )}

              {!image && !showWebcam && (
                <ImageInput onSelectFile={onSelectFile} onWebcamStart={startWebcam} />
              )}

              {data && (
                <div style={{
                  opacity: fadeIn ? 1 : 0,
                  transform: fadeIn ? 'translateY(0)' : 'translateY(16px)',
                  transition: 'opacity 0.4s ease, transform 0.4s ease',
                }}>
                  <PredictionResult
                    data={data}
                    heatmaps={heatmaps}
                    heatmapLoading={heatmapLoading}
                    selectedModel={selectedModel}
                    modelNames={modelNames}
                  />
                </div>
              )}

              {isLoading && (
                <CardContent className={classes.detail}>
                  <div style={{ textAlign: 'center', padding: '20px 0' }}>
                    <CircularProgress size={52} className={classes.loader} thickness={4} />
                    <Typography style={{ marginTop: 20, color: '#1B5E20', fontWeight: 600, fontSize: 16 }}>
                      Analyzing your image...
                    </Typography>
                    <Typography style={{ marginTop: 8, color: '#888', fontSize: 13 }}>
                      Running {selectedModel === "ensemble" ? "all 3 models" : "model prediction"}
                    </Typography>
                  </div>
                </CardContent>
              )}
            </Card>
          </Grid>

          {data && (
            <Grid item xs={12} md={8} lg={6} className={classes.buttonGrid}>
              <div className={classes.actionButtons}>
                <ColorButton
                  variant="contained"
                  className={classes.actionButton}
                  component="span"
                  size="large"
                  onClick={nextImage}
                  startIcon={<Refresh />}
                >
                  New Image
                </ColorButton>

                {hasSaveableResult && (
                  <SecondaryButton
                    variant="contained"
                    className={classes.actionButton}
                    component="span"
                    size="large"
                    onClick={handleSaveToHistory}
                    startIcon={<SaveIcon />}
                    disabled={saved}
                  >
                    {saved ? "✓ Saved!" : "Save to History"}
                  </SecondaryButton>
                )}
              </div>
            </Grid>
          )}
        </Grid>
      </Container>

      <HistoryDialog open={historyDialog} onClose={() => setHistoryDialog(false)} history={history} />
    </React.Fragment>
  );
}
