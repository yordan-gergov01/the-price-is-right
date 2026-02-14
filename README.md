# The Price is Right 🎯💰

A comprehensive machine learning project that predicts product prices using various approaches, from traditional ML to modern deep learning and LLM fine-tuning. This project compares multiple techniques including Random Forest, Neural Networks, and fine-tuned frontier models on Amazon product data.

## 📊 What This Project Does

This project explores and compares different machine learning approaches for product price prediction:

- **Traditional Machine Learning**: Linear Regression, Random Forest with bag-of-words features
- **Deep Learning**: Custom Neural Networks with residual blocks and layer normalization
- **Large Language Models**: Base model prompting and fine-tuned GPT-4.1-nano
- **Human Baseline**: Comparison with human price prediction performance

The system takes product descriptions (title, category, features, details) and predicts their price, evaluating each approach with visual metrics including scatter plots, error trends, and confidence intervals.

## 🔬 Implemented Approaches

### 1. Baseline Models
- **Random Predictor**: Generates random prices (baseline worst-case)
- **Constant Average**: Predicts the training set average price
- **Linear Regression**: Uses simple features (weight, text length)

### 2. Traditional NLP + ML
- **Bag of Words + Linear Regression**: Uses CountVectorizer with 2000 features
- **Random Forest**: Ensemble of 100 decision trees on hashed text features
- Achieves ~$85 average error on test set

### 3. Deep Neural Networks
- **Custom 8-layer Network**: Residual blocks with layer normalization
- **Features**: HashingVectorizer with 5000 binary features
- **Architecture**: Input → 128 → 64^6 → 1 with ReLU activation and dropout
- **Training**: Adam optimizer with cosine annealing scheduler
- **Performance**: ~$75 average error, comparable to human performance

### 4. LLM Fine-tuning
- **Model**: GPT-4.1-nano fine-tuned on pricing task
- **Training Data**: 100-2000 product examples with price labels
- **Approach**: Supervised fine-tuning with batch size 1, 1 epoch
- **Performance**: ~$68 average error (best performance)

## 📊 Dataset

**Source**: Amazon Reviews 2023 (McAuley-Lab/Amazon-Reviews-2023)

**Processing Pipeline**:
1. Filter products by price range ($0.50-$999.49)
2. Extract title, description, features, category, weight
3. Use LLM (GPT-oss-20b via Groq) to create concise summaries
4. Tokenize and prepare prompts for model training

**Dataset Splits**:
- **Full Dataset**: 810K items (800K train, 10K val, 10K test)
- **Lite Dataset**: 22K items (20K train, 1K val, 1K test)

**Data Format**:
```python
Item:
  - title: str
  - category: str
  - price: float
  - summary: str (LLM-generated)
  - weight: float
  - prompt: str (for LLM training)
```

## 🎨 Model Performance Comparison

| Model | Average Error | Description |
|-------|--------------|-------------|
| Random Predictor | ~$300 | Random baseline |
| Constant Average | ~$125 | Training mean |
| Linear Regression | ~$105 | Simple features |
| NLP Linear Regression | ~$95 | Bag of words |
| Random Forest | ~$85 | Ensemble method |
| Deep Neural Network | ~$75 | 8-layer residual network |
| **Human Baseline** | ~$75 | 100 samples |
| **Fine-tuned GPT-4.1-nano** | ~$68 | Best performance |

## 🛠️ Technologies & Tools Used

**Machine Learning**:
- scikit-learn (Random Forest, Linear Regression, CountVectorizer, HashingVectorizer)
- PyTorch (Deep Neural Networks, custom architectures)

**Large Language Models**:
- OpenAI GPT-4.1-nano (fine-tuning)
- Groq API (fast inference for preprocessing)
- LiteLLM (unified LLM interface)

**Data Processing**:
- HuggingFace Datasets (data loading and hosting)
- Pandas, NumPy (data manipulation)
- Pydantic (data validation)

**Visualization & Evaluation**:
- Plotly (interactive scatter plots and error trends)
- Matplotlib (static visualizations)
- Custom evaluation framework with confidence intervals

**Development Tools**:
- Jupyter Notebooks (interactive experimentation)
- python-dotenv (environment management)
- tqdm (progress bars)

## 🔧 Key Components

**Evaluator**: Visual evaluation framework with scatter plots, error trends with confidence intervals, MSE and R² metrics, and color-coded error bands.

**Batch Processing**: Efficient preprocessing using Groq's batch API with automatic file management, job submission, and result fetching.

**Deep Neural Network**: Custom architecture with residual blocks, layer normalization, dropout regularization, and log-scale normalization for price targets.

**Item Model**: Pydantic-based data structure with HuggingFace Hub integration, prompt generation, and token counting utilities.

## 👤 Author

**Yordan Gergov**
- GitHub: [@yordan-gergov01](https://github.com/yordan-gergov01)

---

**Note**: This project is for educational purposes and demonstrates various ML/AI techniques for price prediction.
