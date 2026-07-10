const DISEASE_INFO = {
  "Early Blight": {
    description:
      "Early blight is a common fungal disease caused by Alternaria solani. It affects leaves, stems, and fruits of potato plants.",
    symptoms: [
      "Small, brown to black lesions with concentric rings on older leaves",
      "Yellowing of leaf tissue around the lesions",
      "Lesions may coalesce, causing large areas of leaf tissue to die",
      "Defoliation in severe cases",
    ],
    treatment: [
      "Remove and destroy infected plant debris",
      "Use crop rotation with non-host crops",
      "Apply fungicides (like mancozeb or chlorothalonil) according to label instructions",
      "Ensure good air circulation by proper spacing of plants",
      "Avoid overhead watering",
    ],
  },
  "Late Blight": {
    description:
      "Late blight is a devastating fungal disease caused by Phytophthora infestans. It can destroy entire potato fields in a short time.",
    symptoms: [
      "Dark green to brown, water-soaked lesions on leaves",
      "White, fuzzy growth on the undersides of leaves in wet conditions",
      "Lesions expand rapidly and turn dark brown to black",
      "Brown lesions on stems and tubers",
    ],
    treatment: [
      "Remove and destroy all infected plants immediately",
      "Apply fungicides (like copper-based products or systemic fungicides) as a preventive measure",
      "Use resistant potato varieties",
      "Avoid overhead watering; use drip irrigation instead",
      "Monitor weather conditions; late blight thrives in cool, wet weather",
    ],
  },
  Healthy: {
    description:
      "Your potato plant is healthy! Continue to practice good crop care to keep it that way.",
    symptoms: [
      "Vibrant green leaves with no lesions or discoloration",
      "Strong, upright stems",
      "No visible signs of pests or diseases",
    ],
    treatment: [
      "Maintain a regular watering schedule",
      "Fertilize appropriately for potato plants",
      "Monitor for any signs of pests or diseases",
      "Practice crop rotation",
      "Ensure good soil health",
    ],
  },
};

export default DISEASE_INFO;
