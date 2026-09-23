# TESTING REPORT - Potato Leaf Disease Classifier

**Date:** 2026-08-25  
**Model file (deliverable .pth):** `C:\Users\shadb\Downloads\dataset\deployment\potato_bundle.pth` (385 MB)  
**Prediction method:** soft-vote average of 4 models (EfficientNetV2-B3 + ConvNeXt-Tiny v1 + Swin-Tiny v1 + ConvNeXt-Tiny v2)

## 1. Real-world test: 100 unseen field images (PLD)

| Model | Accuracy /100 |
|---|---|
| Ensemble (4-model) | 42% |
| Swin-Tiny v1 | 42% |
| ConvNeXt-Tiny v1 | 40% |
| ConvNex-Tiny v2 | 39% |
| EfficientNet-B3 v1 | 37% |

### Per-class results (ensemble)

| True class | Correct | Accuracy |
|---|---|---|
| Late Blight | 27/27 | 100% |
| Healthy | 6/15 | 40% |
| Early Blight | 9/58 | 16% |

## 2. Reference test: 100 held-out IPD lab images

| Ensemble & every single model | Accuracy |
|---|---|
| All | **100/100 (100%)** |

## 3. Images that made errors (real-world test)

**Total errors: 58 / 100**

- 44/58 errors: ALL 4 models were wrong together (hard domain-shift cases)
- Main pattern: Early Blight predicted as Late Blight (44 cases), Healthy predicted as Late Blight (9 cases), Early->Healthy (5 cases)

| # | Image filename | True label | Predicted as | Confidence | 4 models correct? (E/C/S/V2) |
|---|---|---|---|---|---|
| 1 | `1692334247300.jpg` | Early Blight | **Late Blight** | 0.6572 | wrong wrong wrong wrong |
| 2 | `1692334247401.jpg` | Early Blight | **Late Blight** | 0.8966 | wrong wrong wrong wrong |
| 3 | `1692335403753.jpg` | Early Blight | **Late Blight** | 0.8341 | wrong wrong wrong wrong |
| 4 | `1692336758769.jpg` | Early Blight | **Late Blight** | 0.9042 | wrong wrong wrong wrong |
| 5 | `20230712_114543.jpg` | Early Blight | **Late Blight** | 0.9785 | wrong wrong wrong wrong |
| 6 | `20230712_122247.jpg` | Early Blight | **Late Blight** | 0.8592 | wrong wrong wrong wrong |
| 7 | `20230712_122911.jpg` | Early Blight | **Late Blight** | 0.9808 | wrong wrong wrong wrong |
| 8 | `20230712_123306.jpg` | Early Blight | **Late Blight** | 0.8988 | wrong wrong wrong wrong |
| 9 | `20230712_154301.jpg` | Early Blight | **Late Blight** | 0.9795 | wrong wrong wrong wrong |
| 10 | `20230712_155554.jpg` | Early Blight | **Late Blight** | 0.9588 | wrong wrong wrong wrong |
| 11 | `20230802_105706.jpg` | Early Blight | **Late Blight** | 0.9799 | wrong wrong wrong wrong |
| 12 | `20230802_105848.jpg` | Early Blight | **Late Blight** | 0.888 | wrong wrong wrong wrong |
| 13 | `20230802_110505.jpg` | Early Blight | **Late Blight** | 0.9816 | wrong wrong wrong wrong |
| 14 | `20230802_110528056.jpg` | Early Blight | **Healthy** | 0.5139 | wrong wrong wrong wrong |
| 15 | `20230802_111326036.jpg` | Early Blight | **Healthy** | 0.7691 | wrong wrong wrong wrong |
| 16 | `20230802_111913.jpg` | Early Blight | **Late Blight** | 0.8236 | wrong wrong wrong wrong |
| 17 | `20230802_112024.jpg` | Early Blight | **Late Blight** | 0.9456 | wrong wrong wrong wrong |
| 18 | `20230802_112343452.jpg` | Early Blight | **Late Blight** | 0.894 | wrong wrong wrong wrong |
| 19 | `20230802_124140776.jpg` | Early Blight | **Healthy** | 0.9058 | wrong wrong wrong wrong |
| 20 | `20230816_122443.jpg` | Early Blight | **Late Blight** | 0.9676 | wrong wrong wrong wrong |
| 21 | `20230816_122510.jpg` | Early Blight | **Late Blight** | 0.9123 | wrong wrong wrong wrong |
| 22 | `IMG_0121.JPG` | Early Blight | **Late Blight** | 0.9763 | wrong wrong wrong wrong |
| 23 | `IMG_0143.JPG` | Early Blight | **Late Blight** | 0.7586 | wrong wrong right wrong |
| 24 | `IMG_0148.JPG` | Early Blight | **Late Blight** | 0.6941 | wrong right wrong wrong |
| 25 | `IMG_0150.JPG` | Early Blight | **Late Blight** | 0.9839 | wrong wrong wrong wrong |
| 26 | `IMG_0253 (1).JPG` | Early Blight | **Late Blight** | 0.9287 | wrong wrong wrong wrong |
| 27 | `IMG_0262.JPG` | Early Blight | **Late Blight** | 0.8982 | wrong wrong wrong wrong |
| 28 | `IMG_0264.JPG` | Early Blight | **Late Blight** | 0.9446 | wrong wrong wrong wrong |
| 29 | `IMG_0727.JPG` | Early Blight | **Late Blight** | 0.4889 | wrong right right wrong |
| 30 | `IMG_0815.JPG` | Early Blight | **Late Blight** | 0.6064 | wrong right wrong right |
| 31 | `IMG_0817.JPG` | Early Blight | **Late Blight** | 0.9755 | wrong wrong wrong wrong |
| 32 | `IMG_0873.JPG` | Early Blight | **Late Blight** | 0.9802 | wrong wrong wrong wrong |
| 33 | `IMG_20230815_112811~2.jpg` | Early Blight | **Healthy** | 0.7496 | wrong wrong wrong right |
| 34 | `IMG_20230815_112858~2.jpg` | Early Blight | **Late Blight** | 0.8345 | wrong wrong wrong right |
| 35 | `IMG_20230815_115653~2.jpg` | Early Blight | **Late Blight** | 0.7433 | wrong wrong wrong wrong |
| 36 | `IMG_20230815_121024~2.jpg` | Early Blight | **Late Blight** | 0.5995 | wrong wrong wrong right |
| 37 | `IMG_20230815_151853.jpg` | Early Blight | **Healthy** | 0.4373 | wrong wrong wrong right |
| 38 | `IMG_20230816_121600.jpg` | Early Blight | **Late Blight** | 0.8525 | wrong wrong wrong wrong |
| 39 | `IMG_4674.JPG` | Early Blight | **Late Blight** | 0.9768 | wrong wrong wrong wrong |
| 40 | `IMG_4725.JPG` | Early Blight | **Late Blight** | 0.9698 | wrong wrong wrong wrong |
| 41 | `IMG_8079.JPG` | Early Blight | **Late Blight** | 0.8546 | wrong wrong wrong wrong |
| 42 | `IMG_8220.JPG` | Early Blight | **Late Blight** | 0.7915 | wrong wrong wrong wrong |
| 43 | `IMG_8245.JPG` | Early Blight | **Late Blight** | 0.9611 | wrong wrong wrong wrong |
| 44 | `IMG_9378.JPG` | Early Blight | **Late Blight** | 0.6232 | wrong wrong wrong right |
| 45 | `IMG_9384.JPG` | Early Blight | **Late Blight** | 0.9673 | wrong wrong wrong wrong |
| 46 | `IMG_9405.JPG` | Early Blight | **Late Blight** | 0.9773 | wrong wrong wrong wrong |
| 47 | `IMG_9406.JPG` | Early Blight | **Late Blight** | 0.9768 | wrong wrong wrong wrong |
| 48 | `IMG_9452.JPG` | Early Blight | **Late Blight** | 0.9735 | wrong wrong wrong wrong |
| 49 | `IMG_9468.JPG` | Early Blight | **Late Blight** | 0.8871 | wrong wrong wrong wrong |
| 50 | `20230712_120140.jpg` | Healthy | **Late Blight** | 0.9607 | wrong wrong wrong wrong |
| 51 | `20230712_121759.jpg` | Healthy | **Late Blight** | 0.9667 | wrong wrong wrong wrong |
| 52 | `20230712_131224.jpg` | Healthy | **Late Blight** | 0.9579 | wrong wrong wrong wrong |
| 53 | `20230712_131450.jpg` | Healthy | **Late Blight** | 0.7295 | wrong wrong right wrong |
| 54 | `20230712_132255.jpg` | Healthy | **Late Blight** | 0.886 | wrong wrong wrong wrong |
| 55 | `20230712_133011.jpg` | Healthy | **Late Blight** | 0.6619 | wrong wrong right wrong |
| 56 | `20230802_123822.jpg` | Healthy | **Late Blight** | 0.5434 | right wrong right wrong |
| 57 | `20230816_121556.jpg` | Healthy | **Late Blight** | 0.6273 | wrong wrong right wrong |
| 58 | `20230922_154136.jpg` | Healthy | **Late Blight** | 0.7314 | wrong wrong right wrong |

## 4. Conclusion

- Same-distribution (lab) images: perfect accuracy.
- Real-world field photos: Late Blight detection excellent (100%), but Early vs Late Blight confusion under uncontrolled conditions drags overall to ~42%.
- Remedy in progress: v3 training on the 37 GB expanded Irish Potato dataset (downloading) to add diversity and close this gap.
