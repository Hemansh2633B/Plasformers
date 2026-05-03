# Plasformers Plugins

Third-party extensions can register:

- `backbone`
- `loss`
- `augmentation`
- `head`
- `exporter`
- `train_runner`
- `evaluator`
- `distill_runner`
- `tune_objective`

Use:

```python
from plas.core.plugins import register_plugin

@register_plugin("loss", "my_loss")
def build_loss(config):
    ...
```
