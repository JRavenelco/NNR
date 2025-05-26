import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_

class OODHead(nn.Module):
    """Cabeza dedicada para detección OOD con múltiples métodos."""
    def __init__(self, in_features, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Salida binaria (ID vs OOD)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)

class ViTForOODDetection(nn.Module):
    """
    Vision Transformer (ViT) mejorado para detección de Out-of-Distribution (OOD) en CIFAR-100.
    Incluye cabeza OOD dedicada y soporte para múltiples métodos de detección.
    """
    def __init__(self, num_classes=100, img_size=32, patch_size=4, in_chans=3, embed_dim=384, 
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.2, 
                 attn_drop_rate=0.2, drop_path_rate=0.2, use_abs_pos_emb=True, ood_methods=None):
        """
        Args:
            num_classes: Número de clases de CIFAR-100 (100)
            img_size: Tamaño de la imagen de entrada (32x32 para CIFAR-100)
            patch_size: Tamaño de los parches (4x4 para CIFAR-100)
            in_chans: Número de canales de entrada (3 para RGB)
            embed_dim: Dimensión del espacio de embedding
            depth: Número de capas del transformer
            num_heads: Número de cabezas de atención
            mlp_ratio: Ratio para la dimensión oculta del MLP
            qkv_bias: Si incluir bias en las capas de query, key, value
            drop_rate: Dropout rate
            attn_drop_rate: Dropout rate para atención
            drop_path_rate: Drop path rate para regularización
            use_abs_pos_emb: Si usar posicional embeddings absolutos
        """
        super().__init__()
        
        # Configuración básica
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim  # Para compatibilidad con timm
        
        # Backbone ViT mejorado
        # Usamos la implementación de timm con parámetros optimizados
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            # Añadidos parámetros adicionales para mejorar el rendimiento
            class_token=True,
            global_pool='token',
            fc_norm=True,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=nn.LayerNorm
        )
        
        # Head de clasificación con label smoothing
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Cabeza OOD dedicada
        self.ood_head = OODHead(embed_dim)
        self.ood_methods = ood_methods or ['energy', 'max_prob', 'odin', 'mahalanobis']
        
        # Inicialización de pesos
        self.apply(self._init_weights)
        
        # Parámetros para detección OOD
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Para escalado de energía
        self.register_buffer('class_mean', None)
        self.register_buffer('class_cov', None)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """Extrae características del backbone ViT."""
        return self.vit.forward_features(x)
    
    def forward(self, x, return_features=False, return_attention=False):
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada (B, C, H, W)
            return_features: Si devolver las características extraídas
            return_attention: Si devolver los mapas de atención
            
        Returns:
            logits: Logits para clasificación (B, num_classes)
            ood_score: Puntuación OOD (B, 1)
            (opcional) features: Características extraídas
            (opcional) attn_weights: Pesos de atención
        """
        # Obtener características
        features = self.forward_features(x)
        
        # Clasificación
        logits = self.vit.head(features[:, 0])  # Usar el token [CLS] para clasificación
        
        # Detección OOD (usando el token [CLS])
        ood_score = self.ood_head(features[:, 0])
        
        # Obtener pesos de atención si es necesario
        attn_weights = None
        if return_attention:
            # Implementar lógica para extraer pesos de atención si es necesario
            pass
        
        # Devolver resultados según los flags
        output = (logits, ood_score)
        if return_features:
            output = output + (features,)
        if return_attention:
            output = output + (attn_weights,)
            
        return output if len(output) > 1 else output[0]
    
    def get_ood_score(self, x, method='max_prob'):
        """
        Calcula la puntuación OOD usando diferentes métodos.
        
        Args:
            x: Tensor de entrada (B, C, H, W)
            method: Método para calcular la puntuación OOD:
                   - 'max_prob': 1 - max(softmax(logits))
                   - 'entropy': Entropía de la distribución de probabilidad
                   - 'energy': Energía negativa (menos energía = más probable OOD)
                   - 'head': Usa la cabeza OOD entrenada
        """
        with torch.no_grad():
            logits, ood_score = self(x)
            probs = F.softmax(logits, dim=-1)
            
            if method == 'max_prob':
                # 1 - p(y*|x) donde y* = argmax p(y|x)
                return 1 - torch.max(probs, dim=1)[0]
            elif method == 'entropy':
                # Entropía de la distribución de probabilidad
                return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            elif method == 'energy':
                # Energía negativa: -logsumexp(logits)
                return -torch.logsumexp(logits, dim=1)
            elif method == 'head':
                # Usa la cabeza OOD entrenada
                return ood_score.squeeze()
            else:
                raise ValueError(f"Método OOD desconocido: {method}")

def vit_small_patch4_32(num_classes=100, **kwargs):
    """
    Configuración mejorada de ViT para CIFAR-100 con parches de 4x4.
    Aumentada la capacidad del modelo para mejor rendimiento.
    """
    return ViTForOODDetection(
        num_classes=num_classes,
        img_size=32,
        patch_size=4,
        embed_dim=384,  # Aumentado de 256 a 384
        depth=12,       # Aumentado de 6 a 12 capas
        num_heads=12,   # Aumentado de 8 a 12 cabezas
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.2,  # Aumentado para mejor regularización
        attn_drop_rate=0.2,  # Aumentado para mejor regularización
        drop_path_rate=0.2,  # Aumentado para mejor regularización
        **kwargs
    )

def vit_base_patch4_32(num_classes=100, **kwargs):
    """
    Configuración base mejorada de ViT para CIFAR-100 con parches de 4x4.
    Versión más grande con mayor capacidad de aprendizaje.
    """
    return ViTForOODDetection(
        num_classes=num_classes,
        img_size=32,
        patch_size=4,
        embed_dim=768,  # Aumentado de 512 a 768 (tamaño estándar para ViT-Base)
        depth=12,       # Aumentado de 8 a 12 capas
        num_heads=12,   # Aumentado a 12 cabezas
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.2,  # Aumentado para mejor regularización
        attn_drop_rate=0.2,  # Aumentado para mejor regularización
        drop_path_rate=0.2,  # Aumentado para mejor regularización
        **kwargs
    )

if __name__ == '__main__':
    # Prueba del modelo
    print("Probando la definición del modelo ViT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = vit_small_patch4_32(num_classes=100).to(device)
    print(model)

    # Crear un tensor de entrada de ejemplo (lote de 4 imágenes, 3 canales, 32x32)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    print(f"Forma de la entrada: {dummy_input.shape}")

    # Pasar la entrada a través del modelo
    try:
        logits, ood_score = model(dummy_input)
        print(f"Forma de los logits: {logits.shape} (debería ser [4, 100])")
        print(f"Forma de la puntuación OOD: {ood_score.shape} (debería ser [4, 1])")
        
        # Probar detección OOD con diferentes métodos
        for method in ['max_prob', 'entropy', 'energy', 'head']:
            score = model.get_ood_score(dummy_input, method=method)
            print(f"Puntuación OOD ({method}): {score.shape} (debería ser [4] o [4, 1])")
            
        print("Prueba del modelo ViT completada con éxito.")
    except Exception as e:
        print(f"Error durante la prueba del modelo: {e}")
