import React from "react";
import CardContent from "@material-ui/core/CardContent";
import Typography from "@material-ui/core/Typography";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import CameraAlt from "@material-ui/icons/CameraAlt";
import CloudUpload from "@material-ui/icons/CloudUpload";
import { DropzoneArea } from "material-ui-dropzone";
import { useStyles } from "../constants/theme";
import { ColorButton } from "../constants/theme";

export default function ImageInput({ onSelectFile, onWebcamStart }) {
  const classes = useStyles();

  return (
    <CardContent className={classes.content} style={{ padding: '24px 28px 28px' }}>
      <div style={{ textAlign: 'center', marginBottom: 20 }}>
        <Typography variant="h6" style={{ color: '#1B5E20', fontWeight: 700, fontSize: 18, marginBottom: 4 }}>
          Upload a Potato Leaf
        </Typography>
        <Typography variant="body2" style={{ color: '#666', fontSize: 14 }}>
          Take a photo or upload an image to diagnose
        </Typography>
      </div>

      <div className={classes.uploadOptions}>
        <ColorButton
          variant="contained"
          onClick={onWebcamStart}
          startIcon={<CameraAlt />}
          style={{ flex: 1, padding: '12px 16px', fontSize: 14 }}
        >
          Take Photo
        </ColorButton>
      </div>

      <DropzoneArea
        acceptedFiles={['image/*']}
        dropzoneText={"Drag & drop a potato leaf image here or tap to browse"}
        onChange={onSelectFile}
        dropzoneClass={classes.dropzone}
        filesLimit={1}
        showFileNames={true}
        showAlerts={false}
        maxFileSize={10000000}
        previewGridProps={{ container: { spacing: 0 } }}
        previewText="Selected image:"
      />

      <div style={{
        marginTop: 24,
        padding: '20px 24px',
        backgroundColor: '#F1F8E9',
        borderRadius: 16,
        border: '1px solid #C8E6C9',
      }}>
        <Typography variant="subtitle2" style={{
          color: '#1B5E20',
          fontWeight: 700,
          marginBottom: 12,
          fontSize: 14,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <CloudUpload style={{ fontSize: 18 }} />
          Tips for Best Results
        </Typography>
        <List dense style={{ padding: 0 }}>
          {[
            'Capture the entire leaf in focus on a plain background',
            'Use natural lighting — avoid harsh shadows or glare',
            'Make sure the leaf fills most of the frame',
            'Include both healthy and diseased areas if present',
          ].map((tip, i) => (
            <ListItem key={i} style={{ padding: '3px 0' }}>
              <ListItemText
                primary={`• ${tip}`}
                primaryTypographyProps={{
                  style: { color: '#33691E', fontSize: 13, lineHeight: 1.5 }
                }}
              />
            </ListItem>
          ))}
        </List>
      </div>
    </CardContent>
  );
}
