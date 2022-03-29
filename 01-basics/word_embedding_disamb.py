# %% [markdown]
# # How do word embeddings eliminate ambiguity?
# 
# I have seen more than one person surprised by the fact that word embedding assigns a unique vector to a word with multiple meanings, as they think that different meanings of a word should be represented by different vectors. In fact, this is exactly the case, except that the process is automatically done by the model.
# 
# In the following, I will demonstrate that at the eighth layer of the BERT model, the word vectors corresponding to the two different meanings of the word 'bank' can be clustered into two groups.

# %%
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from operator import getitem
from scipy.cluster.vq import whiten, kmeans, vq
from sklearn.manifold import MDS
from transformers import BertConfig, BertTokenizer, FlaxBertForSequenceClassification

# %% [markdown]
# We select 10 English sentences that contains the same word 'bank', but with two different meanings.

# %%
dataset = (
    ('He got a large loan from the bank.', 0),
    ('He jumped in and swam to the opposite bank.', 1),
    ('Her bank account was rarely over two hundred.', 0),
    ('She got a bank loan to finance the purchase.', 0),
    ('She waded to the bank and picked up her shoes and stockings.', 1),
    ('Some measures are needed to secure the bank from a flood.', 1),
    ('The bank is increasing the amount they lend to small companies.', 0),
    ('The river burst its left bank after heavy rain.', 1),
    ('We are now in competition with a big foreign bank.', 0),
    ('We pedalled north along the east bank of the river.', 1),
)

# %%
sentences = [x[0] for x in dataset]
labels = [x[1] for x in dataset]

# %% [markdown]
# We load the BERT model from Hugging Face Transformers.

# %%
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = FlaxBertForSequenceClassification.from_pretrained(model_name, config=config)

# %% [markdown]
# We tokenize the sentences.

# %%
inputs = tokenizer(sentences, padding=True, return_tensors='jax')

# %% [markdown]
# We need to find out the absolute positions of the word 'bank' in the tokenized input array, so that we can still retrieve them by those positions after they are processed by the model.

# %%
target_id = tokenizer.convert_tokens_to_ids('bank')
print('target_id:', target_id)

# %%
target_positions = (inputs.input_ids == target_id).argmax(-1)
print('target_positions:', target_positions)

# %% [markdown]
# After we send the input into the model, we can get the hidden states.

# %%
outputs = model(**inputs)
hidden_states = outputs.hidden_states

# %% [markdown]
# We extract the eighth layer.

# %%
nth_layer = 8
hidden_state = hidden_states[nth_layer]

# %% [markdown]
# The hidden state has shape (`n_sents`, `seq_len`, `emb_size`).

# %%
print('hidden_state.shape:', hidden_state.shape)

# %% [markdown]
# We retrieve the embeddings of the word 'bank' by the positions.

# %%
target_embeddings = jax.vmap(getitem, in_axes=(0, 0))(hidden_state, target_positions)

# %%
print('target_embeddings.shape:', target_embeddings.shape)

# %% [markdown]
# Now that we get the embeddings, we can do the _k_-means clustering.

# %%
data = whiten(target_embeddings)
centroids, mean_value = kmeans(data, 2)
clusters, _ = vq(data, centroids)

# %%
acc = np.count_nonzero(labels == clusters).item() / len(sentences)
acc = max(acc, 1. - acc)  # if the labels are exactly the opposite
print('Accuracy:', acc)

# %% [markdown]
# And we can plot the embeddings on a 2D plane after dimensionality reduction.

# %%
mds = MDS(n_components=2)
target_transformed_embeddings = mds.fit_transform(target_embeddings)

# %% [markdown]
# As we can see, the circles and the crosses are assigned different colors. This means that the model outputs are exactly the same as the actual labels.

# %%
xs = target_transformed_embeddings[:, 0]
ys = target_transformed_embeddings[:, 1]
cs = ['red' if cluster == 0 else 'blue' for cluster in clusters]
markers = ['x' if label == 0 else 'o' for label in labels]

# for x, y, c, marker in zip(xs, ys, cs, markers):
#     plt.scatter(x, y, c=c, marker=marker)
# plt.show()


