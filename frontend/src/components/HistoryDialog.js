import React from "react";
import Dialog from "@material-ui/core/Dialog";
import DialogTitle from "@material-ui/core/DialogTitle";
import DialogContent from "@material-ui/core/DialogContent";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";

export default function HistoryDialog({ open, onClose, history }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
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
  );
}
