import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):  #canal de entrada input channel, canal de salida: output channel
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1) 
        # primera capa de convolución, Aplica una convolución 2D sobre una señal de entrada compuesta por 
        # varios planos de entrada. paddingcontrola la cantidad de relleno aplicado a la entrada
        # kernel_size, padding un entero en cuyo caso se usa el mismo valor para la dimensión de alto y ancho
        # dos enteros, en cuyo caso, el primer entero se usa para la dimensión de altura y 
        # el segundo entero para la dimensión de ancho
        self.bn1 = nn.BatchNorm2d(out_c) # Normalización 
        ''''
        La normalización por lotes es una técnica que puede mejorar la tasa de aprendizaje de una red neuronal.
        Lo hace minimizando el cambio de covariable interno, que es esencialmente el fenómeno del cambio de 
        distribución de entrada de cada capa a medida que los parámetros de la capa superior cambian durante 
        el entrenamiento.
        '''

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1) # La entrada es la salida anterior
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU() # Aplica la función de unidad lineal rectificada elemento a elemento:
        '''
        La función de activación es una clase en PyTorch que ayuda a convertir funciones lineales en 
        no lineales y convierte datos complejos en funciones simples para que puedan resolverse fácilmente. 
        Los parámetros no están definidos en la función ReLU y, por lo tanto, no necesitamos usar ReLU 
        como módulo. Cuando tenemos que probar diferentes funciones de activación juntas, es mejor usar 
        init como un módulo y usar todas las funciones de activación en el pase hacia adelante.
        '''

    def forward(self, inputs):
        x = self.conv1(inputs) # convolución
        x = self.bn1(x) # normalización
        x = self.relu(x) # función de activación

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module): # Reducir la imagen a la mitad cada vez
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        # Aplica una agrupación máxima 2D sobre una señal de entrada compuesta por varios planos de entrada.
        # torch.nn.MaxPool2d(kernel_size)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        '''
        Aplica un operador de convolución transpuesta 2D sobre una imagen de entrada compuesta por 
        varios planos de entrada.
        Este módulo se puede ver como el gradiente de Conv2d con respecto a su entrada. También se conoce 
        como convolución de zancada fraccionada o deconvolución (aunque no es una operación de deconvolución 
        real ya que no calcula un verdadero inverso de convolución)
        '''
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64) # comienza por 3 por que son imágenes a color
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    # Devuelve un tensor lleno de números aleatorios de una distribución normal con media 0 
    # y varianza 1 (también llamada distribución normal estándar).
    f = build_unet() # función
    y = f(x) # función 
    print(y.shape)
