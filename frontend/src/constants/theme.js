import { makeStyles, withStyles } from "@material-ui/core/styles";
import { Button } from "@material-ui/core";

// Color palette
const colors = {
  primary: '#2E7D32',
  primaryDark: '#1B5E20',
  primaryLight: '#4CAF50',
  accent: '#FF6F00',
  accentLight: '#FFB300',
  danger: '#C62828',
  dangerLight: '#EF5350',
  warning: '#E65100',
  bg: '#F5F7F4',
  cardBg: '#FFFFFF',
  textPrimary: '#1B1B1B',
  textSecondary: '#5F6368',
  border: '#E0E0E0',
  successBg: '#E8F5E9',
  errorBg: '#FFEBEE',
  warningBg: '#FFF8E1',
};

// Primary action button
export const ColorButton = withStyles((theme) => ({
  root: {
    color: '#FFFFFF',
    backgroundColor: colors.primaryDark,
    '&:hover': {
      backgroundColor: colors.primary,
      transform: 'translateY(-2px)',
      boxShadow: `0 8px 25px rgba(46, 125, 50, 0.35)`,
    },
    borderRadius: 12,
    padding: "14px 28px",
    textTransform: "none",
    fontWeight: 700,
    fontSize: 15,
    letterSpacing: '0.3px',
    transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
    '&:active': {
      transform: 'translateY(0)',
      boxShadow: `0 2px 8px rgba(46, 125, 50, 0.25)`,
    },
  },
}))(Button);

// Secondary action button
export const SecondaryButton = withStyles((theme) => ({
  root: {
    color: '#FFFFFF',
    backgroundColor: colors.primary,
    '&:hover': {
      backgroundColor: colors.primaryDark,
      transform: 'translateY(-2px)',
      boxShadow: `0 8px 25px rgba(46, 125, 50, 0.35)`,
    },
    '&$disabled': {
      backgroundColor: '#A5D6A7',
      color: 'rgba(255,255,255,0.7)',
    },
    borderRadius: 12,
    padding: "14px 28px",
    textTransform: "none",
    fontWeight: 700,
    fontSize: 15,
    letterSpacing: '0.3px',
    transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
  },
  disabled: {},
}))(Button);

export const useStyles = makeStyles((theme) => ({
  grow: { flexGrow: 1 },

  mainContainer: {
    background: `linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 50%, #FFF8E1 100%)`,
    minHeight: "100vh",
    marginTop: 0,
    paddingTop: 32,
    paddingBottom: 48,
  },

  gridContainer: {
    justifyContent: "center",
    padding: "0 16px",
  },

  imageCard: {
    margin: "0 auto",
    maxWidth: 520,
    width: "100%",
    backgroundColor: colors.cardBg,
    boxShadow: '0 4px 24px rgba(0, 0, 0, 0.08)',
    borderRadius: 24,
    overflow: "hidden",
    transition: 'all 0.3s ease',
    '&:hover': {
      boxShadow: '0 8px 40px rgba(0, 0, 0, 0.12)',
    },
  },

  media: {
    width: '100%',
    maxHeight: 400,
    objectFit: 'contain',
    backgroundColor: '#FAFAFA',
    borderRadius: 16,
    margin: '0 auto',
    display: 'block',
  },

  detail: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#FFFFFF',
    '& > *:first-child': {
      marginTop: 0,
    },
  },

  appbar: {
    background: 'linear-gradient(135deg, #1B5E20 0%, #2E7D32 40%, #388E3C 100%)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
    color: '#FFFFFF',
    padding: '6px 0',
  },

  loader: {
    color: colors.primary + ' !important',
  },

  formControl: {
    margin: theme.spacing(2, 0),
    minWidth: 220,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    width: "100%",
    '& .MuiOutlinedInput-root': {
      borderRadius: 12,
      '& fieldset': {
        borderColor: colors.border,
      },
      '&:hover fieldset': {
        borderColor: colors.primary,
      },
      '&.Mui-focused fieldset': {
        borderColor: colors.primary,
      },
    },
  },

  logo: {
    width: 40,
    height: 40,
    borderRadius: 10,
    marginRight: 12,
  },

  dropzone: {
    border: '2px dashed #A5D6A7',
    borderRadius: 16,
    backgroundColor: '#F5FBF5',
    padding: 32,
    minHeight: 180,
    transition: 'all 0.3s ease',
    cursor: 'pointer',
    '&:hover': {
      borderColor: colors.primary,
      backgroundColor: '#EEF7EE',
    },
  },

  uploadOptions: {
    display: 'flex',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 20,
  },

  infoSection: {
    marginTop: 28,
    width: '100%',
    animation: '$fadeInUp 0.4s ease',
  },

  infoTitle: {
    color: colors.primaryDark,
    fontWeight: 700,
    marginBottom: 12,
    fontSize: 18,
    position: 'relative',
    paddingLeft: 0,
  },

  infoList: {
    paddingLeft: 8,
  },

  infoListItem: {
    padding: '6px 0',
    '& .MuiListItemText-root': {
      margin: 0,
    },
  },

  historyButton: {
    color: 'rgba(255,255,255,0.9)',
    '&:hover': {
      backgroundColor: 'rgba(255,255,255,0.1)',
    },
  },

  // Grad-CAM styles
  heatmapSection: {
    marginTop: 24,
    width: '100%',
    borderRadius: 16,
    overflow: 'hidden',
    border: '1px solid #E8F5E9',
    backgroundColor: '#FAFAFA',
    animation: '$fadeInUp 0.4s ease',
  },
  heatmapHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 20px',
    backgroundColor: '#F1F8E9',
    borderBottom: '1px solid #E8F5E9',
    transition: 'background-color 0.2s ease',
    '&:hover': {
      backgroundColor: '#E8F5E9',
    },
  },
  heatmapContent: {
    padding: 20,
  },

  // Unknown image card
  unknownCard: {
    padding: 28,
    textAlign: 'center',
    animation: '$fadeInUp 0.4s ease',
  },

  // Action buttons
  buttonGrid: {
    maxWidth: 480,
    width: '100%',
    marginTop: 20,
  },
  actionButtons: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    width: '100%',
  },
  actionButton: {
    width: '100%',
    borderRadius: 14,
  },

  // Tables
  tableContainer: {
    borderRadius: 16,
    overflow: 'hidden',
    boxShadow: '0 2px 12px rgba(0, 0, 0, 0.06)',
    backgroundColor: '#FFFFFF',
    border: '1px solid #E8F5E9',
  },
  tableHead: {
    backgroundColor: '#E8F5E9',
    '& th': {
      fontWeight: 700,
      color: colors.primaryDark,
      fontSize: 13,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      borderBottom: '2px solid #C8E6C9',
    },
  },
  tableBody: {
    '& td': {
      borderBottom: '1px solid #F0F0F0',
      padding: '14px 16px',
    },
  },

  // Content wrapper
  content: {
    padding: '24px',
  },

  '@keyframes fadeInUp': {
    '0%': { opacity: 0, transform: 'translateY(16px)' },
    '100%': { opacity: 1, transform: 'translateY(0)' },
  },
}));
