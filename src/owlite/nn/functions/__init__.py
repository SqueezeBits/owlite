"""Quantization Aware Training (QAT) is a technique that allows the model to learn the quantization error during the
training process. QAT aims to minimize the loss of accuracy during the quantization process, thus making the model
smaller and faster while maintaining as much of its accuracy as possible. OwLite makes QAT easier, requiring only
minimal changes to an existing training code.

Please review the subdocuments for technical details.

## Usage

To use QAT with OwLite, you can follow your standard training procedure, keeping in mind two aspects:

* QAT is a process that needs to be performed after the
[convert](https://squeezebits.gitbook.io/owlite/python-api/owlite.owlite.owlite/owlite.owlite.convert) stage, where
you have applied the compression configuration in experiment mode using OwLite.&#x20;
* If the optimizer for training was set before calling the convert method, you should set the optimizer again with
the new parameter of the converted mode

Please note that the model converted by OwLite has a fixed batch size. Therefore, you need to set `drop_last=True`
when creating your [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
object.

For example:

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=True, timeout=0,
            worker_init_fn=None, *, prefetch_factor=2,
            persistent_workers=False)
```

This ensures that the DataLoader will discard the last remaining batch if the dataset size is not divisible
by the batch size.

## Tips for Better Results

If you are getting unsatisfactory results from your training, consider adjusting the learning rate or the weight decay.
Lowering the learning rate can help the model converge more smoothly while reducing the weight decay can help prevent
the model from over-fitting.

* **Adjust the Learning Rate**: If the training loss fluctuates, consider reducing the learning rate to stabilize
the training of the compressed model. In this way, the model learns more effectively, leading to better performance.

* **Reduce Weight Decay**: Similarly, if the learning process is fluctuating, consider reducing the weight decay
to stabilize the training of the compressed model. In this way, the model generalizes better for unseen data.
"""
from .clq import clq_function
from .clq_plus import clq_plus_function
from .fake_quantize import FakeQuantizeSignature, fake_quantize
from .ste import ste_function
