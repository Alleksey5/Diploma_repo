import torch
from src.metrics.thop_nets import profile, clever_format
from src.metrics.base_metric import BaseMetric

class THOPMetric(BaseMetric):
    """
    Computes MACs and number of parameters using THOP.
    """

    def __init__(self, input_shape=(1, 1, 16000), verbose=True, name="THOP", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.input_shape = input_shape
        self.verbose = verbose
        self.macs = None
        self.params = None
        self._computed = False  # чтобы не считать повторно

    def update(self, model, inputs=None):
        """
        Computes MACs and Params only once (on first call).
        Args:
            model (nn.Module): the model to evaluate
            inputs (Tensor, optional): unused, for compatibility
        """
        if self._computed:
            return  # уже считали

        model.eval()
        dummy_input = torch.randn(*self.input_shape).to(next(model.parameters()).device)

        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy_input,))
            macs_readable, params_readable = clever_format([macs, params], "%.3f")

        self.macs = macs
        self.params = params
        self.macs_readable = macs_readable
        self.params_readable = params_readable
        self._computed = True

        print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")

    def __call__(self, source=None, predict=None):
        """
        Returns MACs (just to be compatible with MetricTracker).
        """
        return self.macs

    def compute(self):
        return self.macs

    def reset(self):
        self._computed = False
        self.macs = None
        self.params = None
