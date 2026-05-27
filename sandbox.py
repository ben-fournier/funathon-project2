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

# classes les plus courantes
classes_top10 = df["code"].value_counts(sort=True).head(10)
print(classes_top10)

# nombre de classes
n_classes = df["code"].n_unique()
print(n_classes)

# %%
# split dataset

from sklearn.model_selection import train_test_split

target = "code"
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
# X_train = train_df.drop(target)
X_train = train_df["label"]
y_train = train_df[target]

# Valid
# X_valid = valid_df.drop(target)
X_valid = valid_df["label"]
y_valid = valid_df[target]

# Test
# X_test = test_df.drop(target)
X_test = test_df["label"]
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
tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# %%
# Look at an example of tokenization
raw = list(X_train[0:9])
tok_ids = tokenizer.tokenize(raw).input_ids
tokens = [tokenizer.tokenizer.convert_ids_to_tokens(tok_id) for tok_id in tok_ids]
print("Raw text: ", raw)
print("Tokens id:", tok_ids)
print("Tokens:")
for tok in tokens:
    print(tok)

# %%
