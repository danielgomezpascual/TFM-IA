import torch
import torchvision
import torch.nn.functional as F
import PIL
import os

from matplotlib import pyplot
from torch.optim.lr_scheduler import StepLR

'''
Este código implementa una red neuronal U-Net para segmentación de imágenes de inundaciones

Arquitectura del modelo
-----------------------
La U-Net es una arquitectura clásica para segmentación semántica con dos partes:

Encoder (contracción): reduce la resolución espacial con capas Down (MaxPool + doble convolución), extrayendo características progresivamente: 3→64→128→256→512→1024 canales.
Decoder (expansión): reconstruye la resolución con capas Up (upsampling + doble convolución), usando skip connections que concatenan features del encoder con las del decoder.
Salida: una convolución 1×1 que produce 2 canales (inundado / no inundado).

Pipeline de datos
-----------------
Carga imágenes .jpg y sus máscaras .png correspondientes
Redimensiona todo a 100×100 píxeles
Convierte las máscaras a formato one-hot (2 clases binarias)

Entrenamiento
-------------
Optimizador Adam con lr=0.01 y scheduler que reduce el lr a la mitad cada época
Función de pérdida: CrossEntropyLoss
20 épocas de entrenamiento

Métricas y visualización
------------------------
Al terminar genera 4 gráficas:

· Índice de Jaccard (IoU) por época
· Pérdida por época
· Predicción de segmentación
· Imagen original, máscara real y overlay de predicción superpuesta
'''
GPU = True # no puedo usar mi tarjeta GPU porque solo tiene 3 GB mem y el proceso de entrenamiento se queda sin memoria, por lo que utilizo CPU para entrenar el modelo. Si el/la alumn@ tiene una GPU con suficiente memoria, puede cambiar esta variable a True para acelerar el entrenamiento.

device = torch.device('cuda' if torch.cuda.is_available() and GPU else 'cpu') # selección de dispositivo (GPU o CPU). 
    #Si el/la alumn@ tiene un Apple con GPU debe utilizar 'mps' (Metal Performance Shaders) en lugar de 'cuda' 

class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            # Arquitectura de convolución doble
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(torch.nn.Module):

    # U-Net con 4 capas de reducción y 4 capas de expansión, con skip connections entre ellas.
    def __init__(self, n_channels, n_classes, bilinear=False):
        # constructor
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # Codificador
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        # decodificador
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # Forward pass con arquitectura
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Down(torch.nn.Module):
    """Reducción de complejidad con  maxpool luego doble convolución"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Subida de escala luego doble convolución"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # si es bilineal, utilice las convoluciones normales para reducir el número de canales
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x =  self.conv(x)

        x = torch.softmax(x, dim=1)
        return x


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':

    images = os.listdir('data/Image/')
    masks = os.listdir('data/Mask/')

    print(len(images), len(masks))

    image_tensor = list()
    masks_tensor = list()

    # Carga de imágenes y máscaras, preprocesamiento y creación de tensores
    for image in images:
        dd = PIL.Image.open(f'data/Image/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (100, 100))

        tt = tt[None, :, :, :]
        tt = torch.tensor(tt, dtype=torch.float) / 255.

        if tt.shape != (1, 3, 100, 100): # si la imagen no tiene el formato esperado, se omite
            continue

        image_tensor.append(tt)

        # Carga de la máscara correspondiente a la imagen actual
        mask = image.replace('.jpg', '.png')
        dd = PIL.Image.open(f'data/Mask/{mask}')
        mm = torchvision.transforms.functional.pil_to_tensor(dd)
        mm = mm.repeat(3, 1, 1)
        mm = torchvision.transforms.functional.resize(mm, (100, 100))
        mm = mm[:1, :, :]

        mm = torch.tensor((mm > 0.).detach().numpy(), dtype=torch.long)
        mm = torch.nn.functional.one_hot(mm)
        mm = torch.permute(mm, (0, 3, 1, 2))
        mm = torch.tensor(mm, dtype=torch.float)
        
        masks_tensor.append(mm)

    image_tensor = torch.cat(image_tensor)
    print(image_tensor.shape)

    # Concatenar imágenes y máscaras en tensores únicos
    masks_tensor = torch.cat(masks_tensor)
    print(masks_tensor.shape)

    # Instancia de U-Net
    unet = UNet(n_channels=3, n_classes=2).to(device)

    # Creación de dataloaders para imágenes y máscaras
    dataloader_train_image = torch.utils.data.DataLoader(image_tensor, batch_size=16)
    dataloader_train_target = torch.utils.data.DataLoader(masks_tensor, batch_size=16)

    # Optimziador ADAM y scheduler para el aprendizaje
    optim = torch.optim.Adam(unet.parameters(), lr=0.01)
    scheduler = StepLR(optimizer=optim, step_size=1, gamma=0.5)
    cross_entropy = torch.nn.CrossEntropyLoss()

    loss_list = list()
    jaccard_list = list()

    # Bucle de entrenamiento
    for epoch in range(20):
      
        running_loss = 0.
        unet.train() # modo entrenamiento

        jaccard_epoch = list()

        for image, target in zip(dataloader_train_image, dataloader_train_target):
            image = image.to(device)
            target = target.to(device)

            pred = unet(image)
            
            # Cálculo de la pérdida utilizando la función de pérdida de entropía cruzada
            loss = cross_entropy(pred, target)
            running_loss += loss.item()

            loss.backward()
            optim.step()

        for image, target in zip(dataloader_train_image, dataloader_train_target):
            image = image.to(device)
            target = target.to(device)

            pred = unet(image) # predicción de segmentación para el lote actual

            _, pred_unflatten = torch.max(pred, dim=1)
            _, target_unflatten = torch.max(target, dim=1)

            # Cálculo del índice de Jaccard (IoU) para cada imagen en el lote
            intersection = torch.sum(pred_unflatten == target_unflatten, dim=(1, 2))/10000.

            jaccard_epoch.append(torch.mean(intersection).detach().to('cpu'))

        jaccard_list.append(sum(jaccard_epoch)/len(jaccard_epoch))
        loss_list.append(running_loss)
        scheduler.step()
        print(f'Época {epoch} - Pérdida {running_loss} - jaccard {sum(jaccard_epoch)/len(jaccard_epoch)}')

    pyplot.clf()
    pyplot.plot(jaccard_list)
    pyplot.xlabel('Época')
    pyplot.ylabel('Índice de Jaccard')
    pyplot.legend(['Índice de Jaccard'])
    pyplot.title('Índice de Jaccard por Época')
    pyplot.show()

    pyplot.clf()
    pyplot.plot(loss_list)
    pyplot.xlabel('Época')
    pyplot.ylabel('Pérdida')
    pyplot.legend(['Pérdida'])
    pyplot.title('Pérdida por Época')
    pyplot.show()

    pyplot.clf()
    pyplot.imshow(pred_unflatten[2].to(torch.device('cpu')).detach().numpy())
    pyplot.title('Imagen de Predicción')
    
    pyplot.show()

    pyplot.clf()
    pyplot.imshow(torch.permute(image[2].to('cpu'), (1, 2, 0)))
    pyplot.title('Imagen 2 original')
    pyplot.show()
    pyplot.clf()
    pyplot.imshow(target[2][0].to('cpu').detach().numpy())
    pyplot.title('Imagen con máscara real (ground truth)')
    pyplot.show()

    pyplot.clf()
    pyplot.imshow(torch.permute(image[2].to('cpu'), (1, 2, 0)))
    pyplot.imshow(pred_unflatten[2].to('cpu').detach().numpy(), alpha=0.3)
    pyplot.title('Superposición (overlay) de las dos imágenes')
    pyplot.show()