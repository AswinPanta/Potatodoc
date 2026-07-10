import React from "react";
import CardContent from "@material-ui/core/CardContent";
import CameraAlt from "@material-ui/icons/CameraAlt";
import Clear from "@material-ui/icons/Clear";
import { ColorButton } from "../constants/theme";

export default function WebcamCapture({ webcamRef, canvasRef, onCapture, onCancel }) {
  return (
    <CardContent>
      <div style={{ position: 'relative', textAlign: 'center' }}>
        <video ref={webcamRef} autoPlay playsInline style={{ width: '100%', borderRadius: 12 }} />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <div style={{ marginTop: 16, display: 'flex', gap: 12, justifyContent: 'center' }}>
          <ColorButton variant="contained" onClick={onCapture} startIcon={<CameraAlt />}>
            Capture
          </ColorButton>
          <ColorButton variant="contained" onClick={onCancel} startIcon={<Clear />}>
            Cancel
          </ColorButton>
        </div>
      </div>
    </CardContent>
  );
}
