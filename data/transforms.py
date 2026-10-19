"""Image transformation pipelines and noise augmentations."""

import io

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


class AddGaussianNoise:
    """Additive Gaussian noise augmentation.

    Args:
        mean: Mean of the additive noise.
        std: Standard deviation of the additive noise.
    """

    def __init__(self, mean=0.0, std=0.05):
        """Initialize the Gaussian noise transform.

        Args:
            mean: Mean of the additive noise.
            std: Standard deviation of the additive noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Apply additive Gaussian noise.

        Args:
            tensor: Input image tensor.

        Returns:
            Noisy image tensor of the same shape and dtype.
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        """Return a readable description of the transform.

        Returns:
            String representation including the configured mean and std.
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class AddPoissonNoise:
    """Poisson (photon-count) noise augmentation.

    Args:
        scale: Multiplicative factor controlling the noise strength.
    """

    def __init__(self, scale=1.0):
        """Initialize the Poisson noise transform.

        Args:
            scale: Multiplicative factor controlling the noise strength.
        """
        self.scale = scale

    def __call__(self, tensor):
        """Apply Poisson noise to a tensor in the ``[0, 1]`` range.

        Args:
            tensor: Input image tensor with values in ``[0, 1]``.

        Returns:
            Noisy image tensor with values clamped to ``[0, 1]``.
        """
        scaled = tensor * 255.0
        noisy = torch.poisson(scaled * self.scale) / self.scale
        noisy = torch.clamp(noisy, 0, 255) / 255.0
        return noisy

    def __repr__(self):
        """Return a readable description of the transform.

        Returns:
            String representation including the configured scale.
        """
        return f"{self.__class__.__name__}(scale={self.scale})"


class AddSaltPepperNoise:
    """Impulse (salt-and-pepper) noise augmentation.

    Args:
        prob: Total probability of a pixel being replaced by salt or pepper.
    """

    def __init__(self, prob=0.01):
        """Initialize the salt-and-pepper noise transform.

        Args:
            prob: Total probability of a pixel being corrupted.
        """
        self.prob = prob

    def __call__(self, tensor):
        """Apply salt-and-pepper noise in place.

        Args:
            tensor: Input image tensor with values in ``[0, 1]``.

        Returns:
            The modified input tensor with salt (1.0) and pepper (0.0)
            pixels injected.
        """
        salt_pepper = torch.rand_like(tensor)
        tensor[salt_pepper < self.prob / 2] = 0.0
        tensor[salt_pepper > 1 - self.prob / 2] = 1.0
        return tensor

    def __repr__(self):
        """Return a readable description of the transform.

        Returns:
            String representation including the configured probability.
        """
        return f"{self.__class__.__name__}(prob={self.prob})"


class AddSpeckleNoise:
    """Multiplicative speckle noise augmentation.

    Args:
        std: Standard deviation of the multiplicative noise factor.
    """

    def __init__(self, std=0.1):
        """Initialize the speckle noise transform.

        Args:
            std: Standard deviation of the multiplicative noise factor.
        """
        self.std = std

    def __call__(self, tensor):
        """Apply multiplicative speckle noise.

        Args:
            tensor: Input image tensor.

        Returns:
            Noisy image tensor of the same shape and dtype.
        """
        noise = torch.randn_like(tensor) * self.std
        return tensor + tensor * noise

    def __repr__(self):
        """Return a readable description of the transform.

        Returns:
            String representation including the configured std.
        """
        return f"{self.__class__.__name__}(std={self.std})"


class JPEGCompression:
    """JPEG compression artifact simulation.

    Args:
        quality: JPEG quality level in ``[1, 95]``.
    """

    def __init__(self, quality=75):
        """Initialize the JPEG compression transform.

        Args:
            quality: JPEG quality level in ``[1, 95]``.
        """
        self.quality = quality

    def __call__(self, tensor):
        """Apply JPEG compression round-trip.

        Args:
            tensor: Input image tensor with values in ``[0, 1]``.

        Returns:
            Re-encoded image tensor of the same shape and dtype.
        """
        img = transforms.ToPILImage()(tensor)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)
        tensor_jpeg = transforms.ToTensor()(img_jpeg)
        return tensor_jpeg

    def __repr__(self):
        """Return a readable description of the transform.

        Returns:
            String representation including the configured quality.
        """
        return f"{self.__class__.__name__}(quality={self.quality})"


class Blur:
    """Gaussian blur augmentation.

    Args:
        degree: Blur strength; the radius equals ``degree / 10``.
    """

    def __init__(self, degree=10):
        """Initialize the blur transform.

        Args:
            degree: Blur strength; the radius equals ``degree / 10``.
        """
        self.degree = degree

    def __call__(self, tensor):
        """Apply Gaussian blur to the tensor.

        Args:
            tensor: Input image tensor with values in ``[0, 1]``.

        Returns:
            Blurred image tensor of the same shape and dtype.
        """
        img = transforms.ToPILImage()(tensor)
        img = img.filter(ImageFilter.GaussianBlur(radius=self.degree / 10))
        return transforms.ToTensor()(img)

    def __repr__(self):
        """Return a readable description of the transform.

        Returns:
            String representation including the configured degree.
        """
        return f"{self.__class__.__name__}(degree={self.degree})"


def get_transforms(in_height, in_width, mean, std, noise_config=None,
                   train_augment=False):
    """Build train, validation and test transformation pipelines.

    Args:
        in_height: Target image height in pixels.
        in_width: Target image width in pixels.
        mean: Normalization mean (scalar or per-channel sequence).
        std: Normalization standard deviation (scalar or per-channel sequence).
        noise_config: Optional mapping controlling test-time noise. Supported
            keys are ``jpeg_quality``, ``gaussian_std``, ``poisson_scale``,
            ``salt_pepper_prob``, ``speckle_std`` and ``motion_degree``.
            A ``None`` value disables the corresponding transform.
        train_augment: If True, the train pipeline reuses the test-time
            pipeline including configured noise; otherwise train uses the
            plain evaluation pipeline.

    Returns:
        Dictionary with keys ``train``, ``val`` and ``test``, each mapping to
        a ``torchvision.transforms.Compose`` instance.
    """
    eval_pipeline = [
        transforms.Resize((in_height, in_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    test_pipeline = [
        transforms.Resize((in_height, in_width)),
        transforms.ToTensor(),
    ]

    val_transform = transforms.Compose(eval_pipeline)

    test_transforms = list(test_pipeline)

    if noise_config:
        if noise_config.get('jpeg_quality') is not None:
            test_transforms.append(JPEGCompression(quality=noise_config['jpeg_quality']))
        if noise_config.get('gaussian_std') is not None:
            test_transforms.append(AddGaussianNoise(std=noise_config['gaussian_std']))
        if noise_config.get('poisson_scale') is not None:
            test_transforms.append(AddPoissonNoise(scale=noise_config['poisson_scale']))
        if noise_config.get('salt_pepper_prob') is not None:
            test_transforms.append(AddSaltPepperNoise(prob=noise_config['salt_pepper_prob']))
        if noise_config.get('speckle_std') is not None:
            test_transforms.append(AddSpeckleNoise(std=noise_config['speckle_std']))
        if noise_config.get('motion_degree') is not None:
            test_transforms.append(Blur(degree=noise_config['motion_degree']))

    test_transforms.append(transforms.Normalize(mean=mean, std=std))
    test_transform = transforms.Compose(test_transforms)

    if train_augment:
        train_pipeline = test_transforms
    else:
        train_pipeline = list(eval_pipeline)

    train_transform = transforms.Compose(train_pipeline)

    return {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform,
    }