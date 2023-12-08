​![](https://raw.githubusercontent.com/SqueezeBits/owlite/master/.github/images/owlite_logo.png)​​

## OwLite

* OwLite is a low-code AI model compression toolkit for machine learning models.
  * Visualizes computational graphs, identifies bottlenecks, and optimizes latency, and memory usage.
  * Also includes an auto-optimization feature and a device farm management system for evaluating optimized models.

## Key features

#### **AI Model Visualization**

* You can visualize AI models using OwLite's editor function.
  * You can easily understand the structure of the entire model at a glance through GUI,
  * and at the same time, you can easily obtain detailed information about individual nodes.

#### **Quantization by Recommended Setting**

* SqueezeBits' engineers provide recommended quantization settings optimized for the model based on their extensive experience with quantization.
  * This allows you to obtain a lightweight model while minimizing accuracy drop.

#### **Quantization by Custom setting**

* Based on the visualized model, you can apply quantization to each node directly.
  * This allows you to finely adjust the desired performance and optimization.

#### **Latency Benchmark**

* You can perform latency benchmarks within OwLite. This allows you to easily compare existing models and models you have edited, and determine the point at which to download the result.

## **Installation**

To install this pacakage, please use Python 3.10 or higher.

Using pip (Recommended)
```bash
pip install --extra-index-url https://pypi.ngc.nvidia.com git+https://github.com/SqueezeBits/owlite
```

From source
```bash
git clone https://github.com/SqueezeBits/owlite.git
cd owlite
pip install --extra-index-url https://pypi.ngc.nvidia.com -e .
```

## Contact

Please contact [owlite-admin@squeezebits.com](mailto:owlite-admin@squeezebits.com) for any questions or suggestions.

<br>
<br>
<div align="center"><img src="https://raw.githubusercontent.com/SqueezeBits/owlite/master/.github/images/SqueezeBits_orange_H.png" width="300px"></div>
