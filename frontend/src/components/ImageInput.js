import React from "react";
import CardContent from "@material-ui/core/CardContent";
import Typography from "@material-ui/core/Typography";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import CameraAlt from "@material-ui/icons/CameraAlt";
import { DropzoneArea } from "material-ui-dropzone";
import { useStyles } from "../constants/theme";
import { ColorButton } from "../constants/theme";

export default function ImageInput({ onSelectFile, onWebcamStart }) {
  const classes = useStyles();

  return (
    <CardContent className={classes.content} style={{ padding: '16px 24px 24px' }}>
      <div className={classes.uploadOptions}>
        <ColorButton variant="contained" onClick={onWebcamStart} startIcon={<CameraAlt />}>
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

      <div style={{ marginTop: 20, padding: 16, backgroundColor: '#E8F5E9', borderRadius: 12 }}>
        <Typography variant="subtitle2" style={{ color: '#2E7D32', fontWeight: 'bold', marginBottom: 8 }}>
          Tips for Taking Good Photos:
        </Typography>
        <List dense>
          <ListItem><ListItemText primary="• Capture the entire potato leaf in focus" /></ListItem>
          <ListItem><ListItemText primary="• Use good, natural lighting (avoid harsh shadows)" /></ListItem>
          <ListItem><ListItemText primary="• Place the leaf on a plain, neutral background" /></ListItem>
          <ListItem><ListItemText primary="• Make sure the leaf is the main subject of the photo" /></ListItem>
          <ListItem><ListItemText primary="• Capture both healthy and diseased parts if present" /></ListItem>
        </List>
      </div>
    </CardContent>
  );
}
