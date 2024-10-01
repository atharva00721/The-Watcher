import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image


class Model:
    """
    Model class for violence detection using a pre-trained CLIP model.
    Attributes:
        settings (dict): Loaded settings from the provided YAML file.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        model_name (str): Name of the pre-trained model to load.
        threshold (float): Prediction threshold for classifying images.
        model (torch.nn.Module): Loaded CLIP model.
        preprocess (callable): Preprocessing function for images.
        labels (list): List of labels for classification.
        labels_ (list): List of labels with additional text for better accuracy.
        text_features (torch.Tensor): Vectorized text features for labels.
        default_label (str): Default label to use if confidence is below threshold.
    Methods:
        transform_image(image: np.ndarray) -> torch.Tensor:
            Transforms a numpy image array to a tensor suitable for the model.
        tokenize(text: list) -> torch.Tensor:
            Tokenizes a list of text strings into tensors.
        vectorize_text(text: list) -> torch.Tensor:
            Converts a list of text strings into vectorized text features.
        predict_(text_features: torch.Tensor, image_features: torch.Tensor) -> tuple:
            Predicts the top label for the given image features.
        predict(image: np.array) -> dict:
            Predicts the label and confidence for an input image.
        plot_image(image: np.array, title_text: str):
            Plots an image with a title.
    """
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name,
                                                device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = []
        for label in self.labels:
            text = 'a photo of ' + label  # will increase model's accuracy
            self.labels_.append(text)

        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image

    @torch.no_grad()
    def tokenize(self, text: list):
        text = clip.tokenize(text).to(self.device)
        return text

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor,
                 image_features: torch.Tensor):
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        return values, indices

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        '''
        Does prediction on an input image

        Args:
            image (np.array): numpy image with RGB channel ordering type.
                              Don't forget to convert image to RGB if you
                              read images via opencv, otherwise model's accuracy
                              will decrease.

        Returns:
            (dict): dict that contains predictions:
                    {
                    'label': 'some_label',
                    'confidence': 0.X
                    }
                    confidence is calculated based on cosine similarity,
                    thus you may see low conf. values for right predictions.
        '''
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features,
                                        image_features=image_features)
        label_index = indices[0].cpu().item()
        label_text = self.default_label
        model_confidance = abs(values[0].cpu().item())
        if model_confidance >= self.threshold:
            label_text = self.labels[label_index]

        prediction = {
            'label': label_text,
            'confidence': model_confidance
        }

        return prediction

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
