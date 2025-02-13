from argparse import ArgumentParser
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageEnhance, ImageOps

# Data Augmentation Functions
def hflip(image):
    image_aug = ImageOps.mirror(image)
    return image_aug

def brightness(image1, image2, min_factor=0.5, max_factor=1.5):
    enhancer1 = ImageEnhance.Brightness(image1)
    enhancer2 = ImageEnhance.Brightness(image2)
    factor = np.random.uniform(min_factor, max_factor)
    image1_aug = enhancer1.enhance(factor)
    image2_aug = enhancer2.enhance(factor)
    return image1_aug, image2_aug

def contrast(image1, image2, min_factor=0.5, max_factor=1.5):
    enhancer1 = ImageEnhance.Contrast(image1)
    enhancer2 = ImageEnhance.Contrast(image2)
    factor = np.random.uniform(min_factor, max_factor)
    image1_aug = enhancer1.enhance(factor)
    image2_aug = enhancer2.enhance(factor)
    return image1_aug, image2_aug

def rotate(image1, image2, angles=15):
    angle = np.random.uniform(-angles, angles)
    image1_aug = image1.rotate(angle)
    image2_aug = image2.rotate(angle)
    return image1_aug, image2_aug

def augment_images_set(train_images, train_labels, augmentation, augmented_sets):
    indices = np.random.choice(len(train_images), size=augmented_sets, replace=False)
    augmented_dataset = []        
    augmented_labels = []
    for idx in indices:
        pair = train_images[idx]
        label = train_labels[idx]
        image1 = Image.fromarray(pair[:2914].reshape(62, 47) * 255).convert("L")
        image2 = Image.fromarray(pair[2914:].reshape(62, 47) * 255).convert("L")
        if augmentation in [rotate, brightness, contrast]:
            image1_aug, image2_aug = augmentation(image1, image2)
        else:
            image1_aug = augmentation(image1)
            image2_aug = augmentation(image2)
        image1_aug_flat = np.array(image1_aug).flatten()
        image2_aug_flat = np.array(image2_aug).flatten()
        augmented_pair = np.hstack([image1_aug_flat, image2_aug_flat])
        augmented_dataset.append(augmented_pair)
        augmented_labels.append(label)
    augmented_dataset = np.array(augmented_dataset)
    augmented_labels = np.array(augmented_labels)
    return augmented_dataset, augmented_labels

# Feature Engineering function
def pixel_euc_dis(images):
        images1 = images[:, :5828 // 2]
        images2 = images[:, 5828 // 2:]
        pixel_diff = np.abs(images1 - images2)
        euc_dist = np.linalg.norm(images1 - images2, axis=1).reshape(-1, 1)
        features = np.hstack([pixel_diff, euc_dist])
        return features

# Main
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("training_dataset", type=str)
    parser.add_argument("model_filename", type=str)
    args = parser.parse_args()

    # Import Data
    training_dataset = args.training_dataset
    data = joblib.load(training_dataset)
    train_images = data['data']
    train_labels = data['target']
    print('Data Imported')

    # Augmentation
    augmented_dataset_hflip, augmented_labels_hflip = augment_images_set(train_images, train_labels,hflip, 200)
    augmented_dataset_brightness, augmented_labels_brightness = augment_images_set(train_images, train_labels,brightness, 300)
    augmented_dataset_contrast, augmented_labels_contrast = augment_images_set(train_images, train_labels,contrast, 300)
    augmented_dataset_rotate, augmented_labels_rotate = augment_images_set(train_images, train_labels,rotate, 200)

    # Combine images set
    augmented_dataset_combined = np.vstack([
        augmented_dataset_hflip,
        augmented_dataset_brightness,
        augmented_dataset_contrast,
        augmented_dataset_rotate,
    ])
    # Combine labels
    augmented_labels_combined = np.hstack([
        augmented_labels_hflip,
        augmented_labels_brightness,
        augmented_labels_contrast,
        augmented_labels_rotate,
    ])
    print('Images Augmented')
    # Added to original image and label sets
    train_images_aug = np.vstack([train_images, augmented_dataset_combined])
    train_labels_aug = np.hstack([train_labels, augmented_labels_combined])
    print('Added Augmented Samples to Training Set')

    # Final Augmented Images
    train_features_aug = pixel_euc_dis(train_images_aug)
    print('Features Engineered')

    # Preprocessing
    #Standardise
    scaler = StandardScaler()
    train_features_scaled_aug = scaler.fit_transform(train_features_aug)
    #PCA
    pca = PCA(n_components=50)
    train_features_pca_aug = pca.fit_transform(train_features_scaled_aug)
    print('Preprocessing Done')

    #SVM GridSearchCV
    print('Training SVM')
    param_grid_svm = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1],
        }
    svm = SVC(random_state=1)
    grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, verbose=1, n_jobs=-1)
    grid_search_svm.fit(train_features_pca_aug, train_labels_aug)
    print('SVM Training Done')

    # Random Forest GridSearchCV
    print('Training Random Forest')
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [4, 5],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [10, 15, 20]
    }

    rf = RandomForestClassifier(random_state=1)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, verbose=1, n_jobs=-1)
    grid_search_rf.fit(train_features_pca_aug, train_labels_aug)
    print('Random Forest Training Done')

    # Stacking to combine models
    best_rf_params = grid_search_rf.best_params_
    best_svm_params = grid_search_svm.best_params_
    rf_best = RandomForestClassifier(**best_rf_params, random_state=1)
    svm_best = SVC(**best_svm_params, random_state=1)
    meta_classifier = LogisticRegression(random_state=1)
    stacking = StackingClassifier(
        estimators=[
            ('rf', rf_best),
            ('svm', svm_best)
        ],
        final_estimator=meta_classifier,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    # Train the Stacking Classifier
    print("Training Stacking Classifier")
    model = stacking.fit(train_features_pca_aug, train_labels_aug)
    print("Training Stacking Done")

    # Pipeline
    pipeline_model = Pipeline([
        ('pixel_euc_dis', FunctionTransformer(pixel_euc_dis)),
        ('scaler', scaler),
        ('pca', pca),                   
        ('estimator', model)
    ])
    # Save model
    model_name = args.model_filename
    joblib.dump(pipeline_model, model_name)
    print(f"Model saved to {model_name}")




