![OwLite logo](https://github.com/SqueezeBits/owlite/assets/64083281/abaa3ad9-0c86-4a9c-9b8d-f54ed6d9524b)

<div align="center">
<p align="center">
  <a href="https://www.squeezebits.com/">Website</a> •
  <a href="https://owlite.ai/">Web UI</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#contact">Contact</a>
</p>
<p align="center">
  <a href="https://github.com/SqueezeBits/owlite/releases"><img src="https://img.shields.io/github/v/release/SqueezeBits/owlite?color=EE781F" /></a>
  <a href="https://github.com/SqueezeBits/owlite-examples" ><img src="https://img.shields.io/badge/Examples-4BCB7A" /></a>
  <a href="https://squeezebits.gitbook.io/owlite/quick/readme"><img src="https://img.shields.io/badge/Documentation-FFA32C" /></a>

  <a href="https://github.com/SqueezeBits/owlite#installation"><img src="https://img.shields.io/badge/python-~=3.10-blue" /></a>
  <a><img src="https://img.shields.io/badge/pytorch-2.0%20%7C%202.1%20%7C%202.2-blue" /></a>
  <a href="https://github.com/SqueezeBits/owlite/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-APGL--3.0-lightgray" /></a>
</p>    
</div>

## OwLite

* https://owlite.ai
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

To install this package, please use Python 3.10.

Using pip (Recommended)
```bash
pip install owlite --extra-index-url https://pypi.squeezebits.com/
```

From source
```bash
git clone https://github.com/SqueezeBits/owlite.git
cd owlite
pip install .
```

## Getting Started

Please check <a href="https://squeezebits.gitbook.io/owlite/">[OwLite Documentation]</a> for user guide and troubleshooting examples.

Explore <a href="https://github.com/SqueezeBits/owlite-examples/">[OwLite Examples]</a>, a repository showcasing seamless PyTorch model compression into TensorRT engines. Easily integrate OwLite with minimal code changes and explore powerful compression results.

## Contact

Please contact [owlite-admin@squeezebits.com](mailto:owlite-admin@squeezebits.com) for any questions or suggestions.

<br>
<br>
<div align="center"><img src="https://github.com/SqueezeBits/owlite/assets/64083281/bdbf55a6-ead7-42d3-b0b7-f1e8eddfb558" width="300px"></div>