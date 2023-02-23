import oneflow as flow
from flowvision.models import vit_tiny_patch16_224
from flowvision.models import vit_small_patch16_224
from flowvision.models import vit_base_patch16_224
from flowvision.models import vit_large_patch16_224
from flowvision.models import vit_huge_patch14_224
from .register import MODEL


MODEL.register(vit_tiny_patch16_224)
MODEL.register(vit_small_patch16_224)
MODEL.register(vit_base_patch16_224)
MODEL.register(vit_large_patch16_224)
MODEL.register(vit_huge_patch14_224)