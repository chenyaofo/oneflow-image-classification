import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim
import oneflow.cuda as cuda
import oneflow.amp as amp
import oneflow.env as env


class TrainGraph(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        cross_entropy: nn.Module,
        data_loader: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler = None,
        return_pred_and_label=True,
        is_enable_amp: bool = False,
        scale_grad: bool = False,
    ):
        super().__init__()

        self.config.enable_amp(is_enable_amp)

        if is_enable_amp:
            self.set_grad_scaler(
                amp.GradScaler(
                    init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
                )
            )
        if scale_grad:
            self.set_grad_scaler(
                amp.StaticGradScaler(env.get_world_size())
            )

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.world_size = env.get_world_size()
        self.num_devices_per_node = cuda.device_count()
        if self.world_size / self.num_devices_per_node > 1:
            self.config.enable_cudnn_conv_heuristic_search_algo(True)

        self.model = model
        self.cross_entropy = cross_entropy
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.return_pred_and_label = return_pred_and_label

    def build(self):
        samples, targets = self.data_loader()
        samples, targets = samples.to("cuda"), targets.to("cuda")

        outputs: flow.Tensor = self.model(samples)
        loss: flow.Tensor = self.cross_entropy(outputs, targets)

        if self.return_pred_and_label:
            pred = outputs.softmax()
        else:
            pred = None
            targets = None
        loss.backward()
        return loss, pred, targets


class EvalGraph(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        data_loader: nn.Module,
        is_enable_amp: bool = False,
    ):
        super().__init__()
        self.config.enable_amp(is_enable_amp)

        self.config.allow_fuse_add_to_output(True)

        self.data_loader = data_loader
        self.model = model

    def build(self):
        samples, targets = self.data_loader()
        samples, targets = samples.to("cuda"), targets.to("cuda")
        outputs: flow.Tensor = self.model(samples)
        pred = outputs.softmax()
        return pred, targets
