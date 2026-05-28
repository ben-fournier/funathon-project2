# %%
import numpy as np

# Text + categorical data
texts = ["Product description...", "Another product..."]
categorical = np.array([
    [3, 1],  # Product 1: category=3, brand=1
    [5, 2],  # Product 2: category=5, brand=2
])
labels = [0, 1]

# %%
# Load the dataset
import polars as pl

df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)

print(len(df))

target = "code"

# %%
# split dataset

from sklearn.model_selection import train_test_split

train_df, no_train_df = train_test_split(
    df,
    train_size=0.70,
    random_state=11,
    stratify=df[target])
valid_df, test_df = train_test_split(
    no_train_df,
    train_size=0.50,
    random_state=11,
    stratify=no_train_df[target])

# Train
X_train = train_df.drop(target)
# X_train = train_df["label"]
y_train = train_df[target]

# Valid
X_valid = valid_df.drop(target)
# X_valid = valid_df["label"]
y_valid = valid_df[target]

# Test
X_test = test_df.drop(target)
# X_test = test_df["label"]
y_test = test_df[target]


# %%
from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# 1. Create tokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(texts)

# 2. Configure model
model_config = ModelConfig(
    embedding_dim=128,
    num_classes=2,  # Binary classification
)

# 3. Train
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)
training_config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
    raw_labels=False
)
classifier.train(
    X_train=np.array(texts),
    y_train=np.array(labels),
    training_config=training_config
)

# 4. Predict
predictions = classifier.predict(np.array(texts))

# %%

# Configure model with categorical features
model_config_w_categor = ModelConfig(
    embedding_dim=128,
    num_classes=3,
    categorical_vocabulary_sizes=[10, 5],  # 10 categories, 5 brands
    categorical_embedding_dims=[8, 4],
)

# Train
classifier_w_categor = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config_w_categor
)
training_config_w_categor = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
    raw_labels=False,
    raw_categorical_inputs=False,
)
classifier_w_categor.train(
    X_train=np.column_stack([texts, categorical]),
    y_train=np.array(labels),
    training_config=training_config_w_categor
)
# classifier_w_categor.train(
#     X_text=texts,
#     y=labels,
#     X_categorical=categorical,
#     training_config=training_config
# )


# %%
