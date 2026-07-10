import React from "react";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import IconButton from "@material-ui/core/IconButton";
import History from "@material-ui/icons/History";
import { useStyles } from "../constants/theme";

export default function Navbar({ onHistoryClick }) {
  const classes = useStyles();

  return (
    <AppBar position="static" className={classes.appbar} elevation={0}>
      <Toolbar>
        <Avatar src="/icon.png" className={classes.logo} />
        <Typography className={`${classes.grow} ${classes.title}`} variant="h5" noWrap>
          PotatoDoc
        </Typography>
        <IconButton className={classes.historyButton} onClick={onHistoryClick}>
          <History />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}
