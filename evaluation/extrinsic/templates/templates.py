from evaluation.extrinsic.templates.cifar10 import cifar10_classes, cifar10_templates
from evaluation.extrinsic.templates.cifar100 import cifar100_classes, cifar100_templates
from evaluation.extrinsic.templates.imagenet import imagenet_classes, imagenet_templates

DATASET = {
    "cifar10": (cifar10_classes, cifar10_templates),
    "cifar100": (cifar100_classes, cifar100_templates),
    "imagenet": (imagenet_classes, imagenet_templates),
}
