import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image

# ==========================================
# 1. ARQUITETURA U-NET
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Arquitetura U-Net clássica para segmentação semântica,
    com canais reduzidos para evitar estouro de memória (Out of Memory - OOM) na CPU.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.downs = nn.ModuleList([
            DoubleConv(in_channels, 16),
            DoubleConv(16, 32),
            DoubleConv(32, 64),
            DoubleConv(64, 128)
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64),
            nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(64, 32),
            nn.ConvTranspose2d(32, 16, 2, stride=2), DoubleConv(32, 16)
        ])
        
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Down part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Up part
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            
            # Se o tamanho não bater (devido a dimensões ímpares), faz padding (não necessário se 512x512)
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_x)
            
        return self.final_conv(x)

# ==========================================
# 2. DATASET CUSTOMIZADO
# ==========================================
class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        all_images = os.listdir(image_dir)
        self.images = []
        
        # Filtra para treinar APENAS nas imagens que você de fato rotulou na ferramenta.
        # Caso contrário, a IA tenta aprender com 900 imagens em branco e fica viciada em prever "nada".
        for img_name in all_images:
            mask_name = img_name.replace(".jpeg", ".png").replace(".jpg", ".png")
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.images.append(img_name)
                
        print(f"Dataset carregado com {len(self.images)} imagens rotuladas manualmente.")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        # A máscara tem o mesmo nome, mas extensão PNG
        mask_name = self.images[idx].replace(".jpeg", ".png").replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # Retorna máscara vazia (tudo preto) se não houver rotulação
            mask = Image.new("L", image.size, 0)
            
        if self.transform:
            # Transformação conjunta usando as novas APIs do torchvision v2
            image, mask = self.transform(image, mask)
            
        # Garante que a máscara seja 0.0 ou 1.0 e float
        mask = (mask > 0.5).float()
        
        return image, mask

# ==========================================
# 3. PIPELINE DE TREINAMENTO
# ==========================================
def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento da U-Net usando: {device}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_img_dir = os.path.join(base_dir, "data", "processed", "train", "images")
    train_mask_dir = os.path.join(base_dir, "data", "processed", "train", "masks")
    
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(train_img_dir):
        print("Diretório de treino não encontrado. Execute o fragmentador primeiro.")
        return

    # Data Augmentation (Conforme especificado na Metodologia: translações, rotações, contraste)
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(contrast=0.2, brightness=0.2), # Ajustes de contraste
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    dataset = SolarPanelDataset(train_img_dir, train_mask_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Função de perda para classificação binária pixel a pixel
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50 # Aumentado de 10 para 50 para garantir que o modelo aprenda melhor os padrões complexos
    print("Iniciando loop de épocas...")
    for epoch in range(num_epochs):
        model.train()
        epoch_iou, epoch_dice, epoch_prec, epoch_recall = 0.0, 0.0, 0.0, 0.0
        
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Ajuste de dimensão se necessário
            if masks.dim() == 3:
                masks = masks.unsqueeze(1) 
            elif masks.dim() == 5:
                masks = masks.squeeze(1)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Cálculo das Métricas de Segmentação
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                tp = (preds * masks).sum().item()
                fp = (preds * (1 - masks)).sum().item()
                fn = ((1 - preds) * masks).sum().item()
                
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                iou = tp / (tp + fp + fn + 1e-8)
                dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
                
                epoch_iou += iou
                epoch_dice += dice
                epoch_prec += prec
                epoch_recall += rec
            
        # Médias da época
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_iou = epoch_iou / num_batches
        avg_dice = epoch_dice / num_batches
        avg_prec = epoch_prec / num_batches
        avg_rec = epoch_recall / num_batches
        
        print(f"Época [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice/F1: {avg_dice:.4f} | Precisão: {avg_prec:.4f} | Recall: {avg_rec:.4f}")
        
    # Salvamento do modelo treinado
    save_path = os.path.join(models_dir, "unet_solar.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Treinamento finalizado. Modelo salvo em: {save_path}")

if __name__ == "__main__":
    train_unet()
