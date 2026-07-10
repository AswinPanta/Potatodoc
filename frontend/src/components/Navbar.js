import React from "react";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import IconButton from "@material-ui/core/IconButton";
import History from "@material-ui/icons/History";
import LocalFlorist from "@material-ui/icons/LocalFlorist";
import { useStyles } from "../constants/theme";

export default function Navbar({ onHistoryClick }) {
  const classes = useStyles();

  return (
    <AppBar position="static" className={classes.appbar} elevation={0}>
      <Toolbar style={{ maxWidth: 1200, margin: '0 auto', width: '100%', padding: '0 24px' }}>
        <Avatar style={{ backgroundColor: 'rgba(255,255,255,0.2)', width: 40, height: 40, marginRight: 14 }}>
          <LocalFlorist style={{ color: '#FFFFFF', fontSize: 24 }} />
        </Avatar>
        <Typography style={{
          flexGrow: 1,
          fontSize: 26,
          fontWeight: 800,
          letterSpacing: '-0.5px',
          fontFamily: "'Inter', sans-serif",
        }} variant="h5" noWrap>
          PotatoDoc
        </Typography>
        <IconButton
          className={classes.historyButton}
          onClick={onHistoryClick}
          style={{ backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 12, padding: 10 }}
        >
          <History />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}
