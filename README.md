# LBPH Face Recognition

Face recognition with Python and OpenCV. A Haar-Cascade classifier detects the face, and the LBPH (Local Binary Patterns Histograms) algorithm recognizes it. All code is in `impl/Face_Recognition.ipynb`.

## Pipeline

1. Detect faces with Haar-Cascade. Training uses only the largest face in each image; prediction uses every face.
2. Crop it, convert it to grayscale, resize it to 200×200, and apply histogram equalization.
3. Train LBPH on the faces, using each folder name as the label.
4. Report accuracy, a per-class classification report, and a confusion matrix on a validation split and on the held-out test set.
5. Retrain on all training data and save the model.

## Project structure

```
data/
  train/<person name>/*.jpg   # training images (validation split is taken from here)
  test/<person name>/*.jpg    # held-out test images
impl/
  Face_Recognition.ipynb      # all code
result/                       # created automatically
  model/fr_model.yml          # trained LBPH model
  model/labels.json           # label index -> person name
  cascade/                    # Haar XML, downloaded if your OpenCV build lacks it
```

## Setup

```bash
pip install opencv-contrib-python numpy scikit-learn matplotlib jupyter
```

You need `opencv-contrib-python` because it provides `cv2.face`. Plain `opencv-python` does not include it.

## Usage

Open `impl/Face_Recognition.ipynb` and run all cells.

- **Section 2** has every setting (paths, detector parameters, LBPH parameters, face size, split).
- **Section 7** trains, evaluates, and saves the model. Each run overwrites the saved files, so you no longer need to delete them by hand.
- **Section 8** runs `predict_image("path/to/image.jpg")`. It detects and names **every** face in the image, so group photos work. It returns a list of `{name, distance, box}` and draws a green box for known faces and a red box for "Unknown". It loads the saved model, so you don't have to retrain first.
- **Section 9** has the original console menu. Uncomment `main_menu()` to use it.

To add a person, create `data/train/<name>/` and `data/test/<name>/` with their photos, then rerun section 7.

## Results

These results come from the dataset included in the repo (10 people, 249 training images; 22 were skipped because no face was detected):

| Split      | Accuracy      |
|------------|---------------|
| Validation | 80.4% (37/46) |
| Test       | 92.0% (46/50) |

The displayed number is the LBPH distance, and a lower value means a closer match. `unknown_threshold` (in Config) labels faces as "Unknown" when the distance is above the threshold. Because of on this dataset the distances for correct and wrong predictions overlap (roughly 60–115), it is better to set this to None and only set a value if you want to check something out.
