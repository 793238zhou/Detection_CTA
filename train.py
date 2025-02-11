from ultralytics import YOLO
from ultralytics.engine.trainer import Basetrainer

class Train(Basetrainer):
    def __init__(self):
        super().__init__()
        self.num_freeze = 9
        self.freeze = [f'model.{x}.' for x in range(self.num_freeze)]  # layers to freeze 
        model = YOLO("yolov8n.pt") 
        model.add_callback("on_batch_end", self.freeze_layer)

    
    def freeze_layer(self, trainer, kl_threshold=0.1):
        model = trainer.model
        
        if self.kl > kl_threshold:
            num_freeze = 9  
            print(f"KL divergence is {self.kl}, freezing {num_freeze} layers.")
        else:
            num_freeze = 0  
            print(f"KL divergence is {self.kl}, no layers are frozen.")
        
        freeze = [f'model.{x}.' for x in range(num_freeze)]  
        for k, v in model.named_parameters(): 
            v.requires_grad = True  # 默认所有层都可以训练

            # 冻结 Conv 层
            if 'cv1' in k:  
                print(f'freezing Conv cv1 layer: {k}')
                v.requires_grad = False
            if 'cv2' in k:  
                print(f'freezing Conv cv2 layer: {k}')
                v.requires_grad = False

            # 冻结 Adapter 中的第一个 Bottleneck_Adapter 模块
            if 'adapter.m.0' in k: 
                print(f'freezing Adapter first Bottleneck_Adapter layer: {k}')
                v.requires_grad = False

            


Train =    Train()
Train.model.train(data="./dataset.yaml")