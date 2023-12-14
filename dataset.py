from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MultiModalDataset(Dataset):
    def __init__(self, face_image_paths, tongue_image_paths, fundus_image_paths, labels, transform=None):
        self.face_image_paths = face_image_paths
        self.tongue_image_paths = tongue_image_paths
        self.fundus_image_paths = fundus_image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        face_image = Image.open(self.face_image_paths[index])
        tongue_image = Image.open(self.tongue_image_paths[index])
        fundus_image = Image.open(self.fundus_image_paths[index])

        if self.transform:
            face_image = self.transform(face_image)
            tongue_image = self.transform(tongue_image)
            fundus_image = self.transform(fundus_image)

        label = self.labels[index]
        return face_image, tongue_image, fundus_image, label

    def __len__(self):
        return len(self.labels)
