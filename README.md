# Mobile Small Language Model Optimization

A comprehensive framework for optimizing small language models (SLMs) for efficient smartphone deployment through advanced quantization strategies and real-world performance benchmarking.

## 🎯 Project Overview

This project addresses the critical challenge of deploying language models on resource-constrained mobile devices. We systematically evaluate quantization techniques, measure real-world performance across diverse smartphone hardware, and provide evidence-based recommendations for mobile AI deployment.

### Key Research Questions

- How do different quantization strategies affect model quality vs. performance trade-offs?
- What are the optimal deployment configurations for various smartphone hardware tiers?
- Which mobile AI frameworks provide the best balance of performance and compatibility?

## 🔬 Technical Approach

### Target Models

| Model           | Parameters | Use Case                            | Target Devices               |
| --------------- | ---------- | ----------------------------------- | ---------------------------- |
| **SmolLM-135M** | 135M       | Basic text processing, edge cases   | All devices including budget |
| **SmolLM-360M** | 360M       | Balanced performance, general tasks | Mid-range and flagship       |
| **Phi-3-mini**  | 3.8B       | Complex reasoning, premium features | Flagship devices only        |

### Quantization Strategies

#### Post-Training Quantization

- **INT8**: Standard 8-bit integer quantization
- **INT4**: Aggressive 4-bit compression for maximum size reduction
- **FP16**: Half-precision floating point for GPU acceleration

#### Advanced Quantization Methods

- **GPTQ**: Gradient-based post-training quantization with minimal quality loss
- **AWQ**: Activation-aware weight quantization preserving salient weights
- **SmoothQuant**: Smoothing activation outliers for better INT8 performance
- **Mixed Precision**: Layer-wise optimization balancing quality and efficiency

### Mobile Platform Integration

#### Android Deployment

- **TensorFlow Lite** with NNAPI acceleration
- **ONNX Runtime** with GPU/DSP delegate support
- **PyTorch Mobile** for consistent cross-platform deployment

#### iOS Deployment

- **Core ML** with Neural Engine optimization
- **ONNX Runtime** with Metal Performance Shaders
- **TensorFlow Lite** with GPU delegate

## 📱 Device Testing Matrix

### Flagship Tier

- Google Pixel 8 Pro (Tensor G3)
- iPhone 15 Pro (A17 Pro)
- Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3)
- OnePlus 12 (Snapdragon 8 Gen 3)

### Mid-Range Tier

- Google Pixel 7a (Tensor G2)
- iPhone 14 (A15 Bionic)
- Samsung Galaxy A54 (Exynos 1380)

### Budget Tier

- iPhone SE 3rd Gen (A15 Bionic)
- Samsung Galaxy A34 (Dimensity 1080)
- Older Android devices (representative sampling)

## 🏗️ Repository Structure

```
mobile-llm-optimization/
├── src/
│   ├── quantization/           # Model compression pipeline
│   │   ├── post_training/      # INT8, INT4, FP16 quantization
│   │   ├── advanced/           # GPTQ, AWQ, SmoothQuant
│   │   └── mixed_precision/    # Layer-wise optimization
│   ├── mobile/                 # Platform-specific deployment
│   │   ├── android/            # TFLite, ONNX Runtime Android
│   │   ├── ios/                # Core ML, ONNX Runtime iOS
│   │   └── cross_platform/     # PyTorch Mobile
│   ├── benchmarking/           # Performance measurement
│   │   ├── inference_timing/   # Latency profiling
│   │   ├── memory_profiling/   # RAM usage tracking
│   │   └── battery_testing/    # Power consumption analysis
│   └── evaluation/             # Quality assessment
│       ├── benchmarks/         # MMLU, HellaSwag, ARC
│       └── quality_metrics/    # Perplexity, BLEU, human eval
├── mobile_apps/                # Native benchmark applications
│   ├── android/                # Android Studio project
│   └── ios/                    # Xcode project
├── data/                       # Datasets and results
│   ├── models/                 # Original and quantized models
│   ├── benchmarks/             # Performance results
│   └── analysis/               # Trade-off analysis data
├── scripts/                    # Automation and utilities
│   ├── setup/                  # Environment setup
│   ├── training/               # Model preparation
│   └── deployment/             # Mobile deployment automation
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── guides/                 # Usage guides
│   └── results/                # Research findings
└── tests/                      # Test suite
    ├── unit/                   # Unit tests
    └── integration/            # Integration tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA toolkit (for GPU quantization)
- Android Studio (for Android development)
- Xcode (for iOS development)

### Installation

```bash
# Clone the repository
git clone https://github.com/zraicik/mobile-llm-optimization.git
cd mobile-llm-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup mobile development (optional)
./scripts/setup/setup_mobile_dev.sh
```

### Basic Usage

```python
from src.quantization import QuantizationPipeline
from src.evaluation import BenchmarkSuite

# Initialize quantization pipeline
pipeline = QuantizationPipeline(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    quantization_methods=["int8", "int4", "gptq"],
    target_platforms=["android", "ios"]
)

# Run quantization
quantized_models = pipeline.quantize()

# Evaluate performance
benchmark = BenchmarkSuite()
results = benchmark.evaluate(quantized_models)
```

## 📊 Performance Targets

### Compression Goals

- **Size Reduction**: 50-75% model size reduction
- **Quality Retention**: <10% degradation on benchmark scores
- **Speed Improvement**: 2-4x inference speedup

### Mobile Performance Targets

| Device Tier | Max Latency | Peak Memory | Quality Threshold   |
| ----------- | ----------- | ----------- | ------------------- |
| Flagship    | <50ms       | <1GB        | >95% original score |
| Mid-range   | <100ms      | <500MB      | >90% original score |
| Budget      | <200ms      | <300MB      | >85% original score |

## 🔬 Evaluation Methodology

### Quality Benchmarks

- **MMLU**: Massive Multitask Language Understanding
- **HellaSwag**: Commonsense reasoning
- **ARC**: AI2 Reasoning Challenge
- **Custom mobile tasks**: Real-world mobile use cases

### Performance Metrics

- **Inference latency**: Time per token generation
- **Memory usage**: Peak RAM consumption
- **Battery impact**: Power consumption per inference
- **Thermal behavior**: Device heating under load
- **Framework overhead**: Platform-specific optimizations

## 📈 Current Results

_Results will be updated as experiments are completed_

### Model Compression Results

| Model       | Original Size | INT8 Size | INT4 Size | Quality Retention |
| ----------- | ------------- | --------- | --------- | ----------------- |
| SmolLM-135M | 540MB         | 135MB     | 68MB      | TBD               |
| SmolLM-360M | 1.4GB         | 360MB     | 180MB     | TBD               |
| Phi-3-mini  | 15.2GB        | 3.8GB     | 1.9GB     | TBD               |

### Device Performance Summary

_Comprehensive results coming soon_

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- Additional quantization methods
- New mobile platform support
- Extended device compatibility
- Improved benchmarking tools
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting and transformers library
- Microsoft for Phi-3 model family
- Google for TensorFlow Lite and mobile optimization research
- Apple for Core ML framework
- The broader open-source mobile AI community

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{raicik2025mobile,
  title={Mobile Small Language Model Optimization},
  author={Zachary Raicik},
  year={2025},
  url={https://github.com/zraicik/mobile-llm-optimization}
}
```

## 📞 Contact

- **Author**: Zachary Raicik
- **Project Repository**: [mobile-llm-optimization](https://github.com/zraicik/mobile-llm-optimization)
- **Issues**: [GitHub Issues](https://github.com/zraicik/mobile-llm-optimization/issues)

---

**Status**: 🚧 Active Development | **Last Updated**: August 2025
