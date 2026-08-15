import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
import numpy as np

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
    Arquitetura U-Net para segmentação semântica de painéis solares em ortofotos/imagens aéreas,
    com canais otimizados para eficiência computacional e alta fidelidade espacial.
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
        
        # Down part (Encoder)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Up part (Decoder)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            
            # Ajuste de tamanho caso haja discrepância de dimensão
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
                
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_x)
            
        return self.final_conv(x)

# ==========================================
# 2. FUNÇÕES DE PERDA CUSTOMIZADAS (ESTADO DA ARTE)
# ==========================================
class DiceLoss(nn.Module):
    """
    Dice Loss para mitigar o desbalanceamento de classes extremo (painéis solares << fundo).
    Avalia a sobreposição espacial das regiões sem sofrer influência do número de pixels de fundo.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class CompoundSolarLoss(nn.Module):
    """
    Função de Perda Composta: Combina Weighted BCE (wBCE) e Dice Loss.
    - wBCE: Penaliza falsos negativos atribuindo maior peso para a classe minoritária (painel).
    - Dice Loss: Maximiza o coeficiente de sobreposição geométrica entre máscara real e predita.
    """
    def __init__(self, alpha=0.5, beta=0.5, pos_weight=5.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = torch.tensor([pos_weight])
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        # Garante que o pos_weight esteja no mesmo dispositivo
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
            
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        dice = self.dice_loss(logits, targets)
        return self.alpha * bce + self.beta * dice

# ==========================================
# 3. DATASET CUSTOMIZADO
# ==========================================
class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        all_images = os.listdir(image_dir) if os.path.exists(image_dir) else []
        self.images = []
        
        # Filtra para treinar nas imagens que de fato possuem máscara correspondente
        for img_name in all_images:
            if not img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            mask_name = img_name.replace(".jpeg", ".png").replace(".jpg", ".png")
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.images.append(img_name)
                
        print(f"Dataset carregado com {len(self.images)} imagens rotuladas.")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_name = self.images[idx].replace(".jpeg", ".png").replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)
            
        if self.transform:
            image, mask = self.transform(image, mask)
            
        mask = (mask > 0.5).float()
        return image, mask

# ==========================================
# 4. PIPELINE DE TREINAMENTO COM MÉTRICAS FORMAIS
# ==========================================
def train_unet(num_epochs=50, batch_size=2, lr=1e-4, pos_weight=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento da U-Net no dispositivo: {device}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_img_dir = os.path.join(base_dir, "data", "processed", "train", "images")
    train_mask_dir = os.path.join(base_dir, "data", "processed", "train", "masks")
    
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(train_img_dir):
        print(f"Diretório de treino não encontrado em {train_img_dir}. Execute a fragmentação primeiro.")
        return

    # Data Augmentation robusto
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(contrast=0.2, brightness=0.2),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    dataset = SolarPanelDataset(train_img_dir, train_mask_dir, transform=train_transform)
    if len(dataset) == 0:
        print("Aviso: Nenhuma imagem rotulada encontrada. Crie anotações na label_tool.py ou gere pseudo-labels.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Função de Perda Composta (wBCE + Dice)
    criterion = CompoundSolarLoss(alpha=0.5, beta=0.5, pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_dice = 0.0
    save_path = os.path.join(models_dir, "unet_solar.pth")
    best_save_path = os.path.join(models_dir, "unet_solar_best.pth")
    
    print("=" * 70)
    print("INÍCIO DO TREINAMENTO SUPERVISIONADO U-NET (TCC II)")
    print(f"Loss: Compound (wBCE [pos_weight={pos_weight}] + Dice) | Épocas: {num_epochs}")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou, epoch_dice, epoch_prec, epoch_recall = 0.0, 0.0, 0.0, 0.0
        
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            elif masks.dim() == 5:
                masks = masks.squeeze(1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
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
                
        scheduler.step()
        
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_iou = epoch_iou / num_batches
        avg_dice = epoch_dice / num_batches
        avg_prec = epoch_prec / num_batches
        avg_rec = epoch_recall / num_batches
        
        print(f"Época [{epoch+1:02d}/{num_epochs:02d}] | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice (F1): {avg_dice:.4f} | Prec: {avg_prec:.4f} | Rec: {avg_rec:.4f}")
        
        # Salva o melhor modelo
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), best_save_path)
            
    # Salva o checkpoint final
    torch.save(model.state_dict(), save_path)
    print("=" * 70)
    print(f"Treinamento finalizado com sucesso!")
    print(f"Modelo final salvo em: {save_path}")
    print(f"Melhor modelo salvo em: {best_save_path} (Melhor Dice: {best_dice:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    train_unet()
