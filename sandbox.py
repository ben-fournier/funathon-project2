# %%
# If you need to change working directory (default is your interactive .py file location)
import os
os.chdir("/home/onyxia/work/funathon-project2")

# %%
# Import libraries and load environment variables
import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

# %%
# Load the dataset
import polars as pl

df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
print(df.head())
print(len(df))


# %%
# classes à prédire

target = "code"

# classes les plus courantes
classes_top10 = df[target].value_counts(sort=True).head(10)
print(classes_top10)

# nombre de classes
n_classes = df[target].n_unique()
print(n_classes)

# %%
# ajout variable catégorielle

df = df.with_columns([
    pl.col("code").str.slice(0, 1).alias("code_1"),
    pl.col("code").str.slice(0, 2).alias("code_12"),
    pl.col("code").str.slice(3, 1).alias("code_3"),
    pl.col("code").str.slice(2, 3).alias("code_34"),
])

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
# verify split

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test :", X_test.shape)

print("Train:", y_train.value_counts(sort=True, normalize=True))
print("Valid:", y_valid.value_counts(sort=True, normalize=True))
print("Test :", y_test.value_counts(sort=True, normalize=True))

print("Full:", df[target].value_counts(sort=True))
print("Train:", y_train.value_counts(sort=True))
print("Valid:", y_valid.value_counts(sort=True))
print("Test :", y_test.value_counts(sort=True))

# %%
# verify split, all counts
table = (
    df[target].value_counts(sort=True)
    .join(
        y_train.value_counts(sort=True),
        on=target, how="full", coalesce=True,
        suffix="_train")
    .join(
        y_valid.value_counts(sort=True),
        on=target, how="full", coalesce=True,
        suffix="_valid")
    .join(
        y_test.value_counts(sort=True),
        on=target, how="full", coalesce=True,
        suffix="_test")
)

print(table)

# %%
# verify split, all frequencies
table = (
    df[target].value_counts(sort=True)
    .join(
        y_train.value_counts(normalize=True),
        on=target, how="full", coalesce=True,
        suffix="_train")
    .join(
        y_valid.value_counts(normalize=True),
        on=target, how="full", coalesce=True,
        suffix="_valid")
    .join(
        y_test.value_counts(normalize=True),
        on=target, how="full", coalesce=True,
        suffix="_test")
    .sort(by="count",  descending=True)
)

print(table)

# %%
# check missing in train

all_codes = set(df[target])
train_codes = set(y_train)
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")

# %%
# label encoder
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train.to_numpy())

from torchTextClassifiers.value_encoder import ValueEncoder

value_encoder = ValueEncoder(label_encoder=encoder)

# %%
# tokeniser v1 train

from torchTextClassifiers.tokenizers import WordPieceTokenizer

tokenizer = WordPieceTokenizer(vocab_size=1000, output_dim=32)
X_train_text = X_train["label"]
tokenizer.train(X_train_text)

print("Output tensor size:", tokenizer.tokenize(X_train_text[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# %%
# Look at an example of tokenization
raw = list(X_train_text[0:9])
tok_ids = tokenizer.tokenize(raw).input_ids
tokens = [tokenizer.tokenizer.convert_ids_to_tokens(tok_id) for tok_id in tok_ids]
print("Raw text: ", raw)
print("Tokens id:", tok_ids)
print("Tokens:")
for tok in tokens:
    print(tok)

# %%
# model: text + categorical

from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig

X_train

# Text + categorical data
text = "label"
embedding_dim = 128

# Choosing Embedding Dimension
# Task Complexity - Data Size
# Recommended embedding_dim

# Simple (binary) < 1K samples
# 32-64

# Medium (3-5 classes) 1K-10K samples
# 64-128

# Complex (10+ classes) 10K-100K samples
# 128-256

# Very complex > 100K samples
# 256-512

# Configure model with categorical features

categorical = ["code_1", "code_12"]

categorical_vocab_size = [X_train[col].n_unique() for col in categorical]
categorical_embedding_dim = [min(voc_size // 2, 50) for voc_size in categorical_vocab_size]

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,  # df[target].n_unique()
    categorical_vocabulary_sizes=categorical_vocab_size,
    categorical_embedding_dims=categorical_embedding_dim,
)

# build encoders
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(y_train.to_numpy())

cat_encoders = {
    cat: LabelEncoder().fit(
        X_train.get_column(cat)
    )
    for cat in categorical
}

mappings = {
    cat: {
        str(k): int(v)
        for k, v in zip(enc.classes_, enc.transform(enc.classes_))
    }
    for cat, enc in cat_encoders.items()
}
print(mappings)


value_encoders = ValueEncoder(
    label_encoder=label_encoder,
    categorical_encoders=cat_encoders
)

classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoders,
)

# %%
# Train

training_config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    lr=1e-3
)

# classifier.train(
#     X_text=X_train[text],
#     y=y_train,
#     X_categorical=X_train[categorical],
#     training_config=training_config
# )

classifier.train(
    X_train=X_train.select([text] + categorical).to_numpy(),
    y_train=y_train.to_numpy(),
    training_config=training_config
)

predictions = classifier.predict(X_test[text])

# %%
