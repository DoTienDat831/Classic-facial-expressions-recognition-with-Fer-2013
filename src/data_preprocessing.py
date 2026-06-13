import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT,"test"), transform=base_tf)

train_tf_alb = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CoarseDropout(num_holes_range=(1,4), hole_height_range=(1,8), hole_width_range=(1,8), p=0.3),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=0, p=0.4),
    A.Normalize(
    mean=[0.5],
    std=[0.5],
    max_pixel_value=255.0
    ),
    ToTensorV2(),
])

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.base = datasets.ImageFolder(root)
        self.tf = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, label = self.base[i]
        img = np.array(img.convert("L"))   # grayscale np array
        aug = self.tf(image=img)
        return aug["image"], label

train_dataset_aug = AlbumentationsDataset(DATA_ROOT+"train", train_tf_alb)


# Generate extra augmented images for the disgust class and save to disk
disgust_dir = DATA_ROOT + "train/disgust"
imgs = [os.path.join(disgust_dir, f) for f in os.listdir(disgust_dir)]

extra_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.8),
    A.RandomBrightnessContrast(p=0.7),
    A.ElasticTransform(alpha=1, sigma=5, p=0.3),
])

target = 2500  # augment up to this count
import cv2
for idx in range(target - len(imgs)):
    src = imgs[idx % len(imgs)]
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    aug = extra_tf(image=img)["image"]
    out = os.path.join(disgust_dir, f"aug_{idx:04d}.png")
    cv2.imwrite(out, aug)

train_dataset_aug = AlbumentationsDataset(DATA_ROOT+"train", train_tf_alb)


# Build per-sample weights so each class is sampled equally
targets = torch.tensor(train_dataset.targets)
class_counts = torch.bincount(targets)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset_aug,
    batch_size=64,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

classes = train_dataset_aug.base.classes

print("--- COUNTER ---")
counts = Counter(train_dataset_aug.base.targets)
print({classes[k]: v for k, v in counts.items()})

print("\n--- UPDATED WEIGHT ---")
for i in range(len(classes)):
    print(f"{classes[i]:10} : {class_weights[i].item():.6f}")