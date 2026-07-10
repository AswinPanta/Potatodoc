import { makeStyles, withStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

// Primary action button (green/brown)
export const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText("#8B4513"),
    backgroundColor: "#8B4513",
    '&:hover': {
      backgroundColor: "#A0522D",
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(139, 69, 19, 0.3)',
    },
    borderRadius: 12,
    padding: "12px 24px",
    textTransform: "none",
    fontWeight: "bold",
    transition: 'all 0.2s ease',
  },
}))(Button);

// Secondary action button (green)
export const SecondaryButton = withStyles((theme) => ({
  root: {
    color: 'white',
    backgroundColor: "#2E7D32",
    '&:hover': {
      backgroundColor: "#1B5E20",
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(46, 125, 50, 0.3)',
    },
    '&$disabled': {
      backgroundColor: '#A5D6A7',
      color: 'white',
    },
    borderRadius: 12,
    padding: "12px 24px",
    textTransform: "none",
    fontWeight: "bold",
    transition: 'all 0.2s ease',
  },
  disabled: {},
}))(Button);

export const useStyles = makeStyles((theme) => ({
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
    transition: 'box-shadow 0.3s ease',
    '&:hover': {
      boxShadow: '0 12px 48px rgba(0, 0, 0, 0.15)',
    },
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
  actionButtons: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    width: '100%',
  },
  actionButton: {
    width: '100%',
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
    transition: 'border-color 0.3s ease, background-color 0.3s ease',
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
    animation: '$fadeIn 0.3s ease',
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
  // Grad-CAM Heatmap styles
  heatmapSection: {
    marginTop: 24,
    width: '100%',
    backgroundColor: '#FAFAFA',
    borderRadius: 12,
    overflow: 'hidden',
    border: '1px solid #E8F5E9',
  },
  heatmapHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 16px',
    backgroundColor: '#F1F8E9',
    borderBottom: '1px solid #E8F5E9',
    transition: 'background-color 0.2s ease',
    '&:hover': {
      backgroundColor: '#E8F5E9',
    },
  },
  heatmapContent: {
    padding: 16,
  },
  // Unknown image card
  unknownCard: {
    padding: 24,
    textAlign: 'center',
    animation: '$fadeIn 0.3s ease',
  },
  '@keyframes fadeIn': {
    '0%': {
      opacity: 0,
      transform: 'translateY(10px)',
    },
    '100%': {
      opacity: 1,
      transform: 'translateY(0)',
    },
  },
}));
