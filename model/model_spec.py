import torch
import torch.nn.functional as F

def increment_loss(bigger, smaller, margin):
    loss = F.relu(margin + smaller - bigger)
    return loss.sum()

def hinge_loss(bigger, smaller, threshold=0.0):
    loss = F.relu(threshold + smaller - bigger)
    threshold = bigger.mean() - smaller.mean()
    threshold = threshold.detach()
    return loss.mean(), threshold

def clip_loss(similarity):
    caption_loss = F.cross_entropy(similarity, torch.arange(similarity.shape[0], device=similarity.device))
    return caption_loss

class DetailLossCalculator:
    """
    calculate the loss of SPEC
    """
    def __init__(self):
        self.lambda_contrast = 1.0
        self.lambda_details = 1.0 
        self.lambda_neg = 1.0
        self.epsilon = 0.0
        self.beta = 0.0
        self.fix_epsilon = False
        self.fix_beta = False
        
        # record the loss
        self.last_contrastive_loss = 0.0
        self.last_detail_loss = 0.0
        self.last_neg_loss = 0.0
    
    def set_loss_balance(self, lambda_contrast=1.0, lambda_details=1.0, lambda_neg=1.0, epsilon=0.0, beta=0.0):
        self.lambda_contrast = lambda_contrast
        self.lambda_details = lambda_details
        self.lambda_neg = lambda_neg
        self.epsilon = epsilon
        self.beta = beta
        self.fix_epsilon = False if epsilon == 0.0 else True
        self.fix_beta = False if beta == 0.0 else True
    
    def calculate_loss(self, model, images, caption_ids, base_ids, detail_ids, neg_ids):
        """
        calculate the loss of SPEC
        
        Args:
            model: the original LongCLIP model
            images: the image batch
            caption_ids, base_ids, detail_ids, neg_ids: the token ids of four text
            
        Returns:
            the total loss and the component loss
        """
        # get the features
        image_features = model.encode_image(images)
        caption_features = model.encode_text(caption_ids)
        base_features = model.encode_text(base_ids)
        detail_features = model.encode_text(detail_ids)
        neg_features = model.encode_text(neg_ids)
        
        # normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        caption_features = caption_features / caption_features.norm(dim=1, keepdim=True)
        base_features = base_features / base_features.norm(dim=1, keepdim=True)
        detail_features = detail_features / detail_features.norm(dim=1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=1, keepdim=True)
        
        # calculate the similarity
        logit_scale = model.logit_scale.exp()
        logits_caption = logit_scale * torch.matmul(caption_features, image_features.t())
        logits_base = logit_scale * torch.matmul(base_features, image_features.t())
        logits_detail = logit_scale * torch.matmul(detail_features, image_features.t())
        logits_neg = logit_scale * torch.matmul(neg_features, image_features.t())
        
        # calculate the loss
        bs = images.shape[0]
        base_diag = torch.diagonal(logits_base)[:, None]
        detail_diag = torch.diagonal(logits_detail)[:, None]
        neg_diag = torch.diagonal(logits_neg)[:, None]
        
        # contrastive loss
        contrastive = clip_loss(logits_caption)
        
        # detail loss
        if self.fix_epsilon:
            detail = increment_loss(detail_diag, base_diag, self.epsilon) / bs
        else:
            detail, epsilon = hinge_loss(detail_diag, base_diag, self.epsilon)
            self.epsilon = torch.clamp(epsilon, 0, 1.5)
        
        # negative sample loss
        if self.fix_epsilon:
            detail_neg = increment_loss(base_diag, neg_diag, self.epsilon) / bs
        else:
            detail_neg, epsilon = hinge_loss(base_diag, neg_diag, self.epsilon)
            self.epsilon = torch.clamp(epsilon, 0, 1.5)
        
        # total loss
        loss = (
            self.lambda_contrast * contrastive
            + self.lambda_details * detail
            + self.lambda_neg * detail_neg
        )
        
        # record the loss
        self.last_contrastive_loss = (self.lambda_contrast * contrastive).detach().item()
        self.last_detail_loss = (self.lambda_details * detail).detach().item()
        self.last_neg_loss = (self.lambda_neg * detail_neg).detach().item()
        
        return loss, {
            "contrastive": self.lambda_contrast * contrastive,
            "detail": self.lambda_details * detail,
            "neg": self.lambda_neg * detail_neg,
            "epsilon": self.epsilon,
            "beta": self.beta
        }