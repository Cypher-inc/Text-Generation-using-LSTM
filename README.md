
# 🧠 Text Generation using LSTM

This project shows how to train a simple text generation model using Recurrent Neural Networks (RNNs) with Keras. The model learns to predict the next word in a sentence based on the previous words.

---

## 📚 Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
  - [Text Preprocessing](#1-text-preprocessing)
  - [Tokenization](#2-tokenization)
  - [Padding](#3-padding)
  - [Embedding](#4-embedding)
- [Model Architecture](#model-architecture)
- [How to Use](#how-to-use)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Example Output](#example-output)

---

## 📖 Overview

We train an RNN to generate text. It learns by seeing lots of example sentences, then tries to guess the next word one word at a time.

---

## ⚙️ Requirements

```bash
pip install tensorflow numpy
````

---

## 🧠 How It Works

### 1. 📋 Text Preprocessing

We start by cleaning and preparing the text. This might involve:

* Lowercasing
* Removing special characters
* Splitting text into lines or sentences

**Example:**

```text
Original: "Once upon a time, there was a King!"
Cleaned:  "once upon a time there was a king"
```

---

### 2. 🔢 Tokenization

We convert words into numbers because models work with numbers, not text.

**Example:**

```python
tokenizer.word_index = {
    "once": 1,
    "upon": 2,
    "a": 3,
    "time": 4,
    "there": 5,
    "was": 6,
    "king": 7
}

Text: "once upon a time"
Tokens: [1, 2, 3, 4]
```

---

### 3. 🧱 Padding

Neural networks expect all input sequences to be the **same length**. If one sequence is shorter, we **pad** it with zeros (usually at the beginning).

**Example:**

```python
Sequences before padding: [[1, 2, 3], [2, 3, 4, 5]]
Max length: 4

After padding: [[0, 1, 2, 3], [2, 3, 4, 5]]
```

---

### 4. 🎯 Embedding

An **Embedding layer** turns token numbers into dense vectors that represent the meaning of the word in a high-dimensional space.

**Example:**

```python
Input token: 3 (for word "a")
Embedding: [0.14, -0.25, 0.78, ...]  ← vector of floats
```

It helps the model understand how words are related, like "king" and "queen" being close in vector space.

---

## 🧱 Model Architecture

```text
Embedding → LSTM → Dense (softmax)
```

* **Embedding Layer**: Converts tokens to word vectors
* **LSTM Layer**: Learns patterns in sequences
* **Dense Layer**: Outputs probabilities for the next word

---

## ▶️ How to Use

### ✍️ Generate Text

```python
def generateText(seedText, nextWords, maxSequenceLen):
    for _ in range(nextWords):
        tokenList = tokenizer.texts_to_sequences([seedText])[0]
        tokenList = pad_sequences([tokenList], maxlen=maxSequenceLen - 1, padding='pre')
        predicted = rnn.predict(tokenList, verbose=0)
        predictedIndex = np.argmax(predicted)

        outputWord = ''
        for word, index in tokenizer.word_index.items():
            if index == predictedIndex:
                outputWord = word
                break

        seedText += " " + outputWord
    return seedText
```

### 🔁 Run Text Generation

```python
print(generateText("Once upon a time", nextWords=10, maxSequenceLen=max_sequence_len))
```

---

## 💾 Saving and Loading the Model

### ✅ Save the model:

```python
rnn.save('Model1.h5')
```

