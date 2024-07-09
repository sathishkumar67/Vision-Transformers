from modules import *

@dataclass
class Args:
    lr: float = 3e-4
    batch_size: int = 32
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = 128
    device_count: int = 1
        
    def __post_init__(self):
        torch.manual_seed(self.seed)

# instantiate the arguments
args = Args()


train_loss = []
val_loss = []

class VisionTransformer(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        train_loss.append(loss.item())
        self.log("Train_Loss", loss, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=args.lr)
        
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        val_loss.append(loss.item())
        self.log("Val_Loss", loss, prog_bar=True)

        return loss
    
vision_model = vit_b_16(pretrained=True)
vision_model.heads = nn.Linear(768, args.num_class)

vision_transformer = VisionTransformer(vision_model)


