import re
import nltk
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ML & NLP Imports
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# PyTorch & Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

# SpaCy & DisplaCy
import spacy
from spacy import displacy
import os
import nltk

# Set a private user directory for NLTK data to silence permission warnings
nltk_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.append(nltk_dir)

# Now perform NLTK downloads cleanly
nltk.download('stopwords', download_dir=nltk_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_dir, quiet=True)
nltk.download('punkt', download_dir=nltk_dir, quiet=True)

# -----------------------------------------------------------------------------
# NLTK RESOURCE DOWNLOADS
# -----------------------------------------------------------------------------
for resource in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource in ['wordnet', 'omw-1.4'] else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CACHED RESOURCE LOADING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Master NLP & GenAI Academy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 Master NLP Academy")
st.sidebar.markdown("---")

selected_level = st.sidebar.radio(
    "Select Curriculum Module:",
    [
        "🏡 Course Overview",
        "🔤 Text Preprocessing",
        "📊 Vectorization (BoW & TF-IDF)",
        "📐 Embeddings & Cosine Distance",
        "🏷️ Classical ML & POS/NER",
        "🔄 RNNs, LSTMs & Attention",
        "⚡ Transformers & GenAI"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Full Interactive Educational Suite | PyTorch & Streamlit")

# =============================================================================
# COURSE OVERVIEW
# =============================================================================
if selected_level == "🏡 Course Overview":
    st.title("🎓 Interactive Natural Language Processing & GenAI Academy")
    st.markdown("""
    Welcome to the complete, fully interactive educational suite for **Natural Language Processing (NLP) and Generative AI**. 
    This application is designed as an interactive textbook that bridges **core mathematical theory**, **interactive visualizers**, and **runnable Python code examples**.

    ### 📌 What You Will Learn Across the 6 Modules:
    
    1. **🔤 Level 0: Text Preprocessing Basics**  
       *Learn how unstructured text is converted into clean structural tokens using Normalization, Regex Filtering, Stopword Removal, and Stemming vs. Lemmatization.*
    2. **📊 Level 1: Classical Vectorization**  
       *Master numeric representations of documents through Bag-of-Words (BoW), CountVectorizer matrices, and Term Frequency-Inverse Document Frequency (TF-IDF) scaling.*
    3. **📐 Level 2: Dense Embeddings & Vector Space Math**  
       *Explore continuous vector representations (Word2Vec concepts), multi-dimensional semantic space, dot products, and Cosine Distance formulas.*
    4. **🏷️ Level 3: Classical Machine Learning & Linguistic Tagging**  
       *Build statistical classifiers like Naive Bayes, extract grammatical structure via Part-of-Speech (POS) Tagging, and identify entities using Named Entity Recognition (NER).*
    5. **🔄 Level 4: Sequential Deep Learning & Attention**  
       *Understand temporal feedback loops in Recurrent Neural Networks (RNNs), gating equations inside LSTMs, and sequence alignment using Cross-Attention.*
    6. **⚡ Level 5: Modern Transformers & Generative LLMs**  
       *Deconstruct Scaled Dot-Product Self-Attention ($Q, K, V$), inspect Encoder vs. Decoder architectures (BERT vs. GPT), and control generation using Temperature, Top-$k$, and Top-$p$ sampling.*
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Recommended Path:** Work sequentially from Level 0 to Level 5. Use the theoretical breakdowns and runnable code blocks at the bottom of each module to deepen your understanding.")
    with col2:
        st.success("⚡ **Interactive Sandbox:** Every module provides live input controls, real-time matrix recalculations, dynamic Plotly charts, and PyTorch implementations.")

# =============================================================================
# LEVEL 0: TEXT PREPROCESSING
# =============================================================================
elif selected_level == "🔤 Level 0: Text Preprocessing":
    st.title("🔤 Level 0: Text Preprocessing Visualizer")
    st.markdown("Before feeding text to machine learning models, raw strings must be standardized, stripped of noise, and broken down into canonical vocabulary tokens.")

    tab1, tab2 = st.tabs(["⚙️ Cleaning & Tokenization Pipeline", "🌱 Stemming vs. Lemmatization"])

    with tab1:
        st.markdown("""
        ### 📖 Theory & Intuition
        * **Case Normalization:** Converts all text to lowercase to ensure the model treats `"Apple"` and `"apple"` as identical vocabulary indices.
        * **Regex Cleaning:** Strips non-informative artifacts like URLs, social media handles, HTML tags, and punctuation.
        * **Tokenization:** Breaks continuous text strings into discrete atomic units (tokens) like words or subwords.
        * **Stopword Removal:** Eliminates high-frequency grammatical connector words (*"the", "is", "at"*) that carry little domain-specific semantic signal.
        """)
        
        input_text = st.text_area(
            "Enter Raw Input Text:",
            value="The 5 fast runner's were running quickly in NEW YORK city! Check out https://nlp.org #learning",
            height=90
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            do_lower = st.checkbox("Lowercase Text", value=True)
            remove_urls = st.checkbox("Remove URLs & Handles", value=True)
        with col2:
            remove_punct = st.checkbox("Remove Punctuation & Special Chars", value=True)
            remove_digits = st.checkbox("Remove Numbers/Digits", value=False)
        with col3:
            apply_token = st.checkbox("Apply Tokenization", value=True)
            remove_stop = st.checkbox("Remove English Stopwords", value=True)

        processed_text = input_text
        if do_lower:
            processed_text = processed_text.lower()
        if remove_urls:
            processed_text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', processed_text)
        if remove_punct:
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
        if remove_digits:
            processed_text = re.sub(r'\d+', '', processed_text)

        tokens = word_tokenize(processed_text) if apply_token else [processed_text]
        
        if remove_stop and apply_token:
            default_stopwords = {"the", "is", "at", "which", "on", "in", "a", "an", "and", "or", "were", "out", "of"}
            tokens = [t for t in tokens if t.lower() not in default_stopwords]

        st.markdown("---")
        st.subheader("Interactive Pipeline Output")
        
        res_col1, res_col2 = st.columns([1.2, 1])
        with res_col1:
            st.write("**Processed Tokens List:**")
            st.write(tokens)
        with res_col2:
            st.write("**Token Statistics:**")
            st.metric("Total Tokens Remaining", len(tokens))
            st.metric("Unique Tokens", len(set(tokens)))

        st.markdown("---")
        st.subheader("💻 Practical Python Implementation")
        st.code("""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "The 5 fast runner's were running quickly in NEW YORK city! Check out https://nlp.org"

# 1. Regex Normalization
text_clean = raw_text.lower()
text_clean = re.sub(r'https?://\S+|www\.\S+', '', text_clean) # Remove URLs
text_clean = re.sub(r'[^\w\s]', '', text_clean)               # Remove Punctuation

# 2. Tokenization
tokens = word_tokenize(text_clean)

# 3. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]

print("Filtered Tokens:", filtered_tokens)
""", language="python")

    with tab2:
        st.markdown("""
        ### 📖 Stemming vs. Lemmatization
        Both techniques reduce inflectional forms (and sometimes derivationally related forms) of a word to a common base form.
        
        * **Stemming (e.g., Porter Stemmer):** Employs crude algorithmic suffix-stripping heuristics. It cuts off word endings without understanding sentence context or dictionary definitions. This makes it fast, but it frequently generates non-words (e.g., *"running"* $\rightarrow$ *"runni"*).
        * **Lemmatization (e.g., WordNet Lemmatizer):** Performs morphological analysis using a formal dictionary. It maps words back to their canonical base form, known as the **lemma**. It requires knowing the word's **Part-of-Speech (POS)** tag to differentiate between verb forms and noun forms (e.g., *"better"* as a noun vs. adjective).
        """)
        
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        sample_words = st.text_input("Enter space-separated words to compare:", value="running cars feet better wolves studying easily")
        words_list = sample_words.split()
        
        if words_list:
            comp_data = []
            for w in words_list:
                comp_data.append({
                    "Original Word": w,
                    "Porter Stemmer": stemmer.stem(w),
                    "WordNet Lemmatizer (Noun)": lemmatizer.lemmatize(w, pos='n'),
                    "WordNet Lemmatizer (Verb)": lemmatizer.lemmatize(w, pos='v')
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Comparison")
        st.code("""
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "feet", "better", "studying"]

print("WORD         | STEMMED   | LEMMATIZED (Verb)")
print("-" * 45)
for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')
    print(f"{word:<12} | {stem:<9} | {lemma}")
""", language="python")

# =============================================================================
# LEVEL 1: VECTORIZATION
# =============================================================================
elif selected_level == "📊 Level 1: Vectorization (BoW & TF-IDF)":
    st.title("📊 Level 1: Classical Vectorization Visualizer")
    st.markdown("Machine learning algorithms operate on numerical vectors, not raw strings. Vectorization maps text into structured numeric matrices.")

    st.markdown("""
    ### 📖 Mathematical Foundation
    Converting unstructured text documents into fixed-size numeric vectors enables distance calculation and matrix operations across documents.
    """)

    corpus_input = st.text_area(
        "Enter Corpus Documents (One sentence per line):",
        value="I love machine learning and artificial intelligence\nMachine learning is powerful and fun\nArtificial intelligence will shape the future of tech",
        height=110
    )
    
    docs = [d.strip() for d in corpus_input.strip().split("\n") if d.strip()]
    
    if len(docs) > 0:
        tab_bow, tab_tfidf = st.tabs(["🎒 Bag-of-Words (BoW)", "📈 TF-IDF Vectorizer"])
        
        with tab_bow:
            st.markdown("""
            #### Bag-of-Words (BoW) & CountVectorizer
            The Bag-of-Words model ignores word order and grammar, treating each document as an unordered collection ("bag") of words.
            Each column in the resulting matrix represents a unique word from the vocabulary ($V$), and each row represents a document vector filled with raw occurrence counts.
            """)
            
            bow_vec = CountVectorizer()
            bow_matrix = bow_vec.fit_transform(docs)
            
            df_bow = pd.DataFrame(
                bow_matrix.toarray(),
                index=[f"Doc {i+1}" for i in range(len(docs))],
                columns=bow_vec.get_feature_names_out()
            )
            st.dataframe(df_bow.style.highlight_max(axis=1), use_container_width=True)
            
            st.markdown("---")
            st.subheader("💻 Practical Code Example: CountVectorizer")
            st.code("""
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning and artificial intelligence",
    "Machine learning is powerful and fun"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Dense Matrix Representation:\n", bow_matrix.toarray())
""", language="python")

        with tab_tfidf:
            st.markdown("""
            #### Term Frequency-Inverse Document Frequency (TF-IDF)
            While Bag-of-Words records raw counts, common words across all documents can dominate vector values. **TF-IDF** fixes this by weighting term frequency against document frequency.

            **Mathematical Formulation:**
            """)
            st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)")
            st.latex(r"\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in doc } d}{\text{Total terms in doc } d}")
            st.latex(r"\text{IDF}(t, D) = \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)")
            
            st.markdown("""
            * **High TF-IDF score:** Achieved by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents.
            * **Low TF-IDF score:** Assigned to words that appear frequently across nearly all documents (e.g., *"is"*, *"and"*).
            """)
            
            tfidf_vec = TfidfVectorizer()
            tfidf_matrix = tfidf_vec.fit_transform(docs)
            
            df_tfidf = pd.DataFrame(
                tfidf_matrix.toarray(),
                index=[f"Doc {i+1}" for i in range(len(docs))],
                columns=tfidf_vec.get_feature_names_out()
            )
            st.dataframe(df_tfidf.style.background_gradient(cmap="Purples"), use_container_width=True)

            st.markdown("---")
            st.subheader("💻 Practical Code Example: TfidfVectorizer")
            st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = [
    "I love machine learning and artificial intelligence",
    "Machine learning is powerful and fun",
    "Artificial intelligence will shape the future of tech"
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf.get_feature_names_out(),
    index=["Doc 1", "Doc 2", "Doc 3"]
)
print(df.round(3))
""", language="python")

# =============================================================================
# LEVEL 2: EMBEDDINGS
# =============================================================================
elif selected_level == "📐 Level 2: Embeddings & Cosine Distance":
    st.title("📐 Level 2: Dense Embeddings & Vector Space Math")
    st.markdown("Classical vectors (BoW/TF-IDF) suffer from high dimensionality and sparsity, and they fail to capture semantic relationships. **Dense Word Embeddings** map words into low-dimensional continuous space where spatial proximity reflects semantic similarity.")

    st.markdown("""
    ### 📖 Vector Space & Semantic Analogies
    Models like **Word2Vec (Skip-gram / CBOW)** learn vector representations by predicting context words. 
    Because vectors encode concepts, semantic arithmetic becomes possible:
    $$\\mathbf{v}_{\\text{king}} - \\mathbf{v}_{\\text{man}} + \\mathbf{v}_{\\text{woman}} \\approx \\mathbf{v}_{\\text{queen}}$$
    """)
    
    st.subheader("1. Interactive 2D Word Embedding Projection")
    
    embedding_dict = {
        "king": np.array([0.9, 0.85]),
        "queen": np.array([0.85, 0.95]),
        "man": np.array([0.8, 0.2]),
        "woman": np.array([0.75, 0.3]),
        "apple": np.array([-0.8, -0.7]),
        "banana": np.array([-0.7, -0.85]),
        "computer": np.array([-0.2, 0.7]),
        "laptop": np.array([-0.15, 0.75])
    }
    
    df_embed = pd.DataFrame(
        [{"Word": k, "Dim 1": v[0], "Dim 2": v[1]} for k, v in embedding_dict.items()]
    )
    
    fig_embed = px.scatter(
        df_embed, x="Dim 1", y="Dim 2", text="Word", size_max=60,
        title="2D Semantic Vector Space Visualization"
    )
    # Fixed CSS color parameter to avoid Plotly validation exception
    fig_embed.update_traces(textposition='top center', marker=dict(size=12, color='rebeccapurple'))
    st.plotly_chart(fig_embed, use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. Cosine Similarity vs. Euclidean Distance")
    st.markdown("""
    * **Cosine Similarity:** Measures the cosine of the angle $\\theta$ between two vectors. It ranges from $-1$ (opposite) to $1$ (identical direction), ignoring magnitude differences.
    * **Euclidean Distance:** Measures the straight-line distance between two point coordinates in geometric space.
    """)
    
    st.latex(r"\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}")
    
    c1, c2 = st.columns(2)
    with c1:
        word1 = st.selectbox("Select Word A:", list(embedding_dict.keys()), index=0)
    with c2:
        word2 = st.selectbox("Select Word B:", list(embedding_dict.keys()), index=1)
        
    vec1 = embedding_dict[word1].reshape(1, -1)
    vec2 = embedding_dict[word2].reshape(1, -1)
    
    sim_score = cosine_similarity(vec1, vec2)[0][0]
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.markdown(f"### Cosine Similarity ($\\cos \\theta$): `{sim_score:.4f}`")
        st.progress(max(0.0, float(sim_score)))
    with col_metric2:
        st.markdown(f"### Euclidean Distance ($d$): `{euclidean_dist:.4f}`")
        
    st.markdown("---")
    st.subheader("💻 Practical Code Example: PyTorch Embeddings & Cosine Distance")
    st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define an Embedding Layer (Vocabulary Size = 10, Embedding Dimension = 4)
embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=4)

# 2. Look up embedding vectors for Word Index 1 ("king") and Word Index 2 ("queen")
idx_king = torch.tensor([1])
idx_queen = torch.tensor([2])

vec_king = embedding_layer(idx_king)
vec_queen = embedding_layer(idx_queen)

# 3. Compute Cosine Similarity
similarity = F.cosine_similarity(vec_king, vec_queen)

print("King Vector:", vec_king.detach().numpy())
print("Queen Vector:", vec_queen.detach().numpy())
print("Cosine Similarity:", similarity.item())
""", language="python")

# =============================================================================
# LEVEL 3: CLASSICAL ML & POS/NER
# =============================================================================
elif selected_level == "🏷️ Level 3: Classical ML & POS/NER":
    st.title("🏷️ Level 3: Classical ML & Linguistic Tagging")
    st.markdown("Supervised statistical machine learning models combine vocabulary features with probabilistic models to classify text and extract linguistic structure.")
    
    tab1, tab2, tab3 = st.tabs([
        "🧮 1. Naive Bayes Classifier", 
        "🏷️ 2. Part-of-Speech (POS) Tagging", 
        "🔍 3. Named Entity Recognition (NER)"
    ])
    
    with tab1:
        st.header("1. Multinomial Naive Bayes Sentiment Classifier")
        st.markdown("""
        ### 📖 Theory & Bayes Theorem
        Naive Bayes relies on Bayes' Theorem to calculate the posterior probability of a category $y$ (e.g., *Positive* vs. *Negative*) given a set of input feature words $x_1, x_2, \dots, x_n$.
        It makes a strong ("naive") assumption that all word features are conditionally independent given the class label.
        """)
        
        st.latex(r"P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \dots, x_n)}")
        
        default_training_data = [
            ("I love this movie, it is fantastic and great!", "Positive"),
            ("Awesome storyline and brilliant acting!", "Positive"),
            ("Superb experience, highly recommended!", "Positive"),
            ("Terrible film, absolute waste of time.", "Negative"),
            ("Horrible acting and boring script.", "Negative"),
            ("Bad plot, worst movie I have ever seen.", "Negative")
        ]
        
        st.subheader("Training Dataset Sample")
        df_train = pd.DataFrame(default_training_data, columns=["Text", "Sentiment"])
        st.dataframe(df_train, use_container_width=True)
        
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(df_train["Text"])
        y_train = df_train["Sentiment"]
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        st.subheader("Interactive Sentiment Inference")
        test_sentence = st.text_input(
            "Enter a review sentence to classify:", 
            value="The acting was brilliant and fantastic, but plot was boring."
        )
        
        if test_sentence:
            X_test = vectorizer.transform([test_sentence])
            prediction = model.predict(X_test)[0]
            probabilities = model.predict_proba(X_test)[0]
            classes = model.classes_
            
            col_res, col_chart = st.columns(2)
            
            with col_res:
                if prediction == "Positive":
                    st.success(f"### Predicted Sentiment: **{prediction}** 🎉")
                else:
                    st.error(f"### Predicted Sentiment: **{prediction}** 😞")
                    
                df_probs = pd.DataFrame({"Class": classes, "Probability": probabilities})
                st.table(df_probs.style.format({"Probability": "{:.2%}"}))
                
            with col_chart:
                fig_prob = px.bar(
                    df_probs, x="Class", y="Probability", color="Class",
                    color_discrete_map={"Positive": "green", "Negative": "red"},
                    title="Class Posterior Probabilities"
                )
                fig_prob.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Example: Scikit-Learn Naive Bayes Pipeline")
        st.code("""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Training Corpus
train_texts = [
    "I love this movie, fantastic acting",
    "Terrible film, waste of time and horrible script"
]
labels = ["Positive", "Negative"]

# 2. Build Pipeline (CountVectorizer -> MultinomialNB)
model_pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
model_pipeline.fit(train_texts, labels)

# 3. Predict on New Sentence
test_text = ["The acting was fantastic"]
prediction = model_pipeline.predict(test_text)
probabilities = model_pipeline.predict_proba(test_text)

print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {dict(zip(model_pipeline.classes_, probabilities[0]))}")
""", language="python")

    with tab2:
        st.header("2. Part-of-Speech (POS) Tagging")
        st.markdown("""
        ### 📖 Theory & Penn Treebank Conventions
        POS tagging marks up words in a text corpus as corresponding to a specific part of speech (nouns, verbs, adjectives, adverbs) based on both its definition and its surrounding context in the sentence.
        """)
        
        pos_sample_text = st.text_input(
            "Enter a sentence for POS tagging:",
            value="The quick brown fox jumps gracefully over the lazy dog."
        )
        
        if pos_sample_text:
            tokens = word_tokenize(pos_sample_text)
            tagged_words = pos_tag(tokens)
            
            df_pos = pd.DataFrame(tagged_words, columns=["Word / Token", "POS Tag"])
            tag_meanings = {
                "NN": "Noun (singular)", "NNS": "Noun (plural)", "NNP": "Proper Noun",
                "VB": "Verb (base form)", "VBD": "Verb (past tense)", "VBZ": "Verb (3rd person singular)",
                "JJ": "Adjective", "RB": "Adverb", "DT": "Determiner", "IN": "Preposition/Conjunction"
            }
            df_pos["Description"] = df_pos["POS Tag"].map(lambda tag: tag_meanings.get(tag, "Other Grammatical Role"))
            
            col_pos_table, col_pos_viz = st.columns([1.2, 1])
            with col_pos_table:
                st.subheader("Tagged Output Matrix")
                st.dataframe(df_pos, use_container_width=True)
            with col_pos_viz:
                st.subheader("POS Tag Distribution")
                tag_counts = df_pos["POS Tag"].value_counts().reset_index()
                tag_counts.columns = ["Tag", "Count"]
                fig_pos = px.pie(tag_counts, values="Count", names="Tag", title="Grammar Breakdown")
                st.plotly_chart(fig_pos, use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Example: NLTK POS Tagging")
        st.code("""
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

text = "The quick brown fox jumps gracefully over the lazy dog."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

print("POS Tagged Output:\n", tagged)
""", language="python")

    with tab3:
        st.header("3. Named Entity Recognition (NER)")
        st.markdown("""
        ### 📖 Theory & Information Extraction
        NER locates and classifies named entities mentioned in unstructured text into predefined categories such as person names, organizations, locations, monetary values, percentages, and temporal expressions.
        """)
        
        ner_sample_text = st.text_area(
            "Enter document text for NER extraction:",
            value="Elon Musk announced that Tesla will invest $5 billion to build a new Gigafactory in Berlin by December 2026. Microsoft and Google expressed interest in collaborating.",
            height=90
        )
        
        if ner_sample_text:
            doc = nlp(ner_sample_text)
            st.subheader("Interactive Entity Highlight View")
            html_ner = displacy.render(doc, style="ent", page=False)
            st.markdown(f'<div style="background-color: #f9f9f9; color: #111; padding: 20px; border-radius: 8px; line-height: 2.2;">{html_ner}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Extracted Entities Breakdown")
            entities_data = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]
            if entities_data:
                df_ner = pd.DataFrame(entities_data, columns=["Entity Text", "Entity Label", "Category Description"])
                st.dataframe(df_ner, use_container_width=True)
            else:
                st.info("No recognized named entities found in the text.")

        st.markdown("---")
        st.subheader("💻 Practical Code Example: SpaCy Entity Extraction")
        st.code("""
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Elon Musk announced that Tesla will invest $5 billion in Berlin by December 2026."

doc = nlp(text)

print("Extracted Entities:")
for ent in doc.ents:
    print(f"Entity: {ent.text:<15} | Label: {ent.label_:<10} | Meaning: {spacy.explain(ent.label_)}")
""", language="python")

# =============================================================================
# LEVEL 4: RNNS, LSTMS & ATTENTION
# =============================================================================
if selected_level == "🔄 Level 4: RNNs, LSTMs & Attention":
    st.title("🔄 Level 4: Deep Learning for Sequences & Attention")
    st.markdown("""
    Unlike classical ML models that treat inputs independently, sequential deep learning models process ordered text or time-series data by maintaining 
    an internal state (**recurrent memory**) that persists information across time steps.
    """)

    tab1, tab2, tab3 = st.tabs([
        "🔁 1. RNN Hidden State Flow", 
        "🚪 2. LSTM Gating Engine", 
        "👁️ 3. Attention Mechanism Matrix"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: RNN Mechanics
    # -------------------------------------------------------------------------
    with tab1:
        st.header("1. Recurrent Neural Network (RNN) Mechanics")
        
        st.info("""
        💡 **Core Intuition:** A Vanilla RNN acts like a conveyor belt processing text word-by-word. At each step $t$, 
        it combines what it's seeing *now* ($x_t$) with its memory of what came *before* ($h_{t-1}$) to form a new memory ($h_t$).
        """)

        st.markdown("### 📖 Mathematical Equations")
        st.latex(r"h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)")

        col_input, col_sim = st.columns([1, 1])
        with col_input:
            sequence_text = st.text_input("Sentence for unrolled RNN steps:", value="The movie was unexpectedly awesome")
            hidden_dim = st.slider("Hidden State Dimension ($h_t$ size):", 2, 8, 4)
            st.caption("Larger hidden dimensions allow the memory vector to track more nuanced semantic concepts simultaneously.")

        words = sequence_text.strip().split()
        if words:
            np.random.seed(42)
            vocab = list(set(words))
            embed_dim = 4
            word_embeddings = {word: np.random.randn(embed_dim) for word in vocab}
            
            W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.5
            W_xh = np.random.randn(hidden_dim, embed_dim) * 0.5
            b_h = np.zeros(hidden_dim)
            
            hidden_states = []
            h_prev = np.zeros(hidden_dim)
            
            for t, word in enumerate(words):
                x_t = word_embeddings[word]
                h_t = np.tanh(np.dot(W_hh, h_prev) + np.dot(W_xh, x_t) + b_h)
                hidden_states.append(h_t)
                h_prev = h_t
                
            df_h = pd.DataFrame(
                hidden_states, 
                index=[f"t={i+1}: '{w}'" for i, w in enumerate(words)],
                columns=[f"h[{i}]" for i in range(hidden_dim)]
            )
            
            with col_sim:
                st.subheader("Hidden Memory Values ($h_t$) Across Steps")
                st.dataframe(df_h.style.background_gradient(cmap="Purples"), use_container_width=True)
                st.caption("Notice how earlier words influence later hidden states, but their influence dampens over time.")

            fig_rnn = go.Figure()
            for dim_idx in range(hidden_dim):
                fig_rnn.add_trace(go.Scatter(
                    x=[f"Step {i+1}\n('{w}')" for i, w in enumerate(words)],
                    y=[h[dim_idx] for h in hidden_states],
                    mode='lines+markers',
                    name=f'Memory Neuron h[{dim_idx}]'
                ))
            fig_rnn.update_layout(
                title="Activation Trajectory of Hidden Memory Dimensions", 
                xaxis_title="Input Time Steps",
                yaxis_title="Neuron Activation [-1 to +1]",
                yaxis=dict(range=[-1.1, 1.1])
            )
            st.plotly_chart(fig_rnn, use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Example: PyTorch Vanilla RNN")
        st.code("""
import torch
import torch.nn as nn

# Setup: Batch Size=1, Sequence Length=5, Input Feature Dim=4, Hidden Memory Dim=8
batch_size, seq_len, input_dim, hidden_dim = 1, 5, 4, 8

# Instantiate PyTorch Vanilla RNN Layer
rnn_layer = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

# Dummy Input Tensor representing 5 sequential token embeddings
x = torch.randn(batch_size, seq_len, input_dim)

# Forward Pass (processes entire sequence)
output_seq, h_final = rnn_layer(x)

print("Output Sequence Shape (Hidden State at EVERY step):", output_seq.shape) # [1, 5, 8]
print("Final Hidden Memory State Shape (Last step only):", h_final.shape)       # [1, 1, 8]
""", language="python")

    # -------------------------------------------------------------------------
    # TAB 2: LSTM Gating Engine
    # -------------------------------------------------------------------------
    with tab2:
        st.header("2. Long Short-Term Memory (LSTM) Gating Engine")
        
        st.warning("""
        ⚠️ **The Problem with Vanilla RNNs:** As sequences get long, multiplying weight matrices repeatedly causes gradients to either explode or drop to zero (**Vanishing Gradient Problem**). Vanilla RNNs effectively forget words seen 10+ steps ago.
        """)

        st.success("""
        💡 **The LSTM Solution:** LSTMs separate the architecture into two pathways:
        1. **Cell State ($C_t$):** An uninterrupted "long-term memory highway" that allows gradients to flow linearly without vanishing.
        2. **Hidden State ($h_t$):** The "short-term working memory" filtered through selective gating.
        """)

        st.markdown("### 📖 Step-by-Step Gate Operations")
        st.latex(r"\text{1. Forget Gate: } f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")
        st.latex(r"\text{2. Input Gate: } i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{and Candidate Memory: } \tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)")
        st.latex(r"\text{3. Cell Update: } C_t = (f_t \odot C_{t-1}) + (i_t \odot \tilde{C}_t)")
        st.latex(r"\text{4. Output Gate: } o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{and Hidden State: } h_t = o_t \odot \tanh(C_t)")

        st.markdown("---")
        st.subheader("🎛️ Interactive Gate Simulation Engine")
        st.caption("Adjust the gate values below to simulate how an LSTM updates its long-term memory cell state ($C_t$) and outputs working memory ($h_t$).")

        col_gate_controls, col_gate_viz = st.columns([1, 1])
        
        with col_gate_controls:
            forget_val = st.slider("Forget Gate Activation ($f_t$):", 0.0, 1.0, 0.9, help="0 = Wipe memory clean | 1 = Retain everything")
            input_val = st.slider("Input Gate Activation ($i_t$):", 0.0, 1.0, 0.7, help="0 = Ignore new input | 1 = Fully accept new input")
            candidate_val = st.slider("Candidate New Memory ($\tilde{C}_t$):", -1.0, 1.0, 0.5, help="Information extracted from current word")
            prev_cell = st.slider("Previous Cell Memory ($C_{t-1}$):", -5.0, 5.0, 2.0, help="Existing long-term context stored from prior steps")
            output_val = st.slider("Output Gate Activation ($o_t$):", 0.0, 1.0, 0.8, help="Controls how much long-term memory to expose to short-term working state")

        retained_memory = forget_val * prev_cell
        added_memory = input_val * candidate_val
        new_cell_state = retained_memory + added_memory
        new_hidden_state = output_val * np.tanh(new_cell_state)
        
        with col_gate_viz:
            st.markdown("#### 🧮 Computation Breakdown")
            st.latex(rf"C_t = ({forget_val} \times {prev_cell}) + ({input_val} \times {candidate_val})")
            st.write(f"* **Retained Prior Memory:** `{retained_memory:.4f}`")
            st.write(f"* **Newly Added Memory:** `{added_memory:.4f}`")
            st.markdown(f"### 👉 **Updated Long-Term Cell State ($C_t$):** `{new_cell_state:.4f}`")
            
            st.markdown("---")
            st.latex(rf"h_t = {output_val} \times \tanh({new_cell_state:.4f})")
            st.markdown(f"### 👉 **Updated Working Hidden State ($h_t$):** `{new_hidden_state:.4f}`")

        st.markdown("---")
        st.subheader("💻 Practical Code Example: PyTorch LSTM Layer Execution")
        st.code("""
import torch
import torch.nn as nn

# Sequence Parameters: Batch=1, Sequence Length=4, Feature Dim=10, Hidden/Cell Dim=16
batch_size, seq_len, input_dim, hidden_dim = 1, 4, 10, 16

lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
dummy_input = torch.randn(batch_size, seq_len, input_dim)

# Forward pass returns full hidden outputs along with final (h_n, c_n) tuples
output_seq, (h_n, c_n) = lstm(dummy_input)

print("Full Sequence Output Shape:", output_seq.shape) # [1, 4, 16]
print("Final Short-Term State (h_n):", h_n.shape)     # [1, 1, 16]
print("Final Long-Term Cell Memory (c_n):", c_n.shape) # [1, 1, 16]
""", language="python")

    # -------------------------------------------------------------------------
    # TAB 3: Sequence-to-Sequence Attention
    # -------------------------------------------------------------------------
    with tab3:
        st.header("3. Sequence-to-Sequence Cross-Attention Mechanism")
        
        st.info("""
        💡 **Core Intuition:** Early translation models compressed an entire source sentence into a single vector before generating text. This created an bottleneck for long sentences.
        **Attention** gives the decoder a "spotlight" to dynamically scan all encoder hidden states at every step of text generation.
        """)

        st.latex(r"\text{Attention Weight } \alpha_{i,j} = \frac{\exp(\text{score}(s_{i-1}, h_j))}{\sum_k \exp(\text{score}(s_{i-1}, h_k))}")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            source_sentence = st.text_input("Source Sentence (French):", value="Le chat noir dort sur le tapis")
        with col_s2:
            target_sentence = st.text_input("Target Sentence (English):", value="The black cat sleeps on the mat")
            
        src_words = source_sentence.strip().split()
        tgt_words = target_sentence.strip().split()
        
        if src_words and tgt_words:
            np.random.seed(100)
            raw_scores = np.random.randn(len(tgt_words), len(src_words)) * 0.5
            
            # Introduce realistic alignment heuristics
            for i, t in enumerate(tgt_words):
                for j, s in enumerate(src_words):
                    if t.lower() == s.lower():
                        raw_scores[i, j] += 3.0
                    elif (t.lower() == "black" and s.lower() == "noir") or (t.lower() == "cat" and s.lower() == "chat"):
                        raw_scores[i, j] += 4.0
                    elif (t.lower() == "the" and s.lower() in ["le", "la"]):
                        raw_scores[i, j] += 3.5
                        
            attn_matrix = np.exp(raw_scores) / np.sum(np.exp(raw_scores), axis=1, keepdims=True)
            df_attn = pd.DataFrame(attn_matrix, index=tgt_words, columns=src_words)
            
            st.subheader("Cross-Attention Heatmap (Alignment Weights)")
            st.caption("Each row represents a target token, showing how much focus it places on each source word during translation.")
            fig_attn = px.imshow(
                df_attn, 
                labels=dict(x="Source Tokens (French)", y="Target Tokens (English)", color="Attention Weight"),
                x=src_words, y=tgt_words, color_continuous_scale="Viridis", text_auto=".2f"
            )
            st.plotly_chart(fig_attn, use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Example: Softmax Attention Alignment Matrix")
        st.code("""
import torch
import torch.nn.functional as F

# Query (Decoder hidden states): Shape [Target_Len, Dim] = [3, 8]
queries = torch.randn(3, 8)
# Keys (Encoder hidden states): Shape [Source_Len, Dim] = [4, 8]
keys = torch.randn(4, 8)

# 1. Compute Dot-Product Alignment Scores between all Queries and Keys
scores = torch.matmul(queries, keys.T) # Shape [3, 4]

# 2. Softmax along source sequence dimension to construct probability weights
attention_weights = F.softmax(scores, dim=-1)

print("Attention Alignment Weights (Rows sum to 1.0):\n", attention_weights.detach().numpy().round(3))
""", language="python")


# =============================================================================
# LEVEL 5: TRANSFORMERS & GENAI
# =============================================================================
elif selected_level == "⚡ Level 5: Transformers & GenAI":
    st.title("⚡ Level 5: Modern Transformers & Generative LLMs")
    st.markdown("""
    Transformers replaced step-by-step recurrence with **Self-Attention**. By processing all sequence tokens concurrently in parallel, 
    Transformers scale far more efficiently on GPUs and capture relationships across long text contexts.
    """)

    tab1, tab2, tab3 = st.tabs([
        "🎯 1. Self-Attention Math (Q, K, V)", 
        "🤖 2. BERT (Masked LM) vs. GPT (Causal LM)", 
        "🎛️ 3. LLM Generation Parameters"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Scaled Dot-Product Self-Attention
    # -------------------------------------------------------------------------
    with tab1:
        st.header("1. Scaled Dot-Product Self-Attention ($Q, K, V$)")
        
        st.info("""
        💡 **Database Analogy for Q, K, V:**
        * **Query ($Q$):** The search parameter typed into a search engine (What the current word is looking for).
        * **Key ($K$):** The title/tags of every document in the database (What features each word offers).
        * **Value ($V$):** The actual contents of the documents returned (The semantic representation passed along).
        """)

        st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V")
        st.caption("Scaling by $\\sqrt{d_k}$ prevents dot products from growing excessively large in high dimensions, which would cause softmax gradients to vanish.")

        seq_input = st.text_input("Sentence for Self-Attention:", value="The bank of the river")
        tokens = seq_input.strip().split()
        
        if tokens:
            N = len(tokens)
            d_k = 4
            np.random.seed(42)
            
            # Projections
            Q = np.random.randn(N, d_k)
            K = np.random.randn(N, d_k)
            V = np.random.randn(N, d_k)
            
            # Step by step calculations
            scores = np.dot(Q, K.T)
            scaled_scores = scores / np.sqrt(d_k)
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            context_vectors = np.dot(attn_weights, V)
            
            st.markdown("### 🔍 Step 1: Linear Projections ($Q, K, V$)")
            col_q, col_k, col_v = st.columns(3)
            with col_q:
                st.write("**Queries ($Q = X W_Q$)**")
                st.dataframe(pd.DataFrame(Q, index=tokens, columns=[f"q_{i}" for i in range(d_k)]))
            with col_k:
                st.write("**Keys ($K = X W_K$)**")
                st.dataframe(pd.DataFrame(K, index=tokens, columns=[f"k_{i}" for i in range(d_k)]))
            with col_v:
                st.write("**Values ($V = X W_V$)**")
                st.dataframe(pd.DataFrame(V, index=tokens, columns=[f"v_{i}" for i in range(d_k)]))

            st.markdown("---")
            st.markdown("### 🔍 Step 2 & 3: Softmax Attention Weights & Output Context")
            col_attn, col_out = st.columns([1.2, 1])
            with col_attn:
                st.subheader("Attention Weight Matrix")
                df_attn = pd.DataFrame(attn_weights, index=tokens, columns=tokens)
                fig_attn = px.imshow(df_attn, x=tokens, y=tokens, color_continuous_scale="Purples", text_auto=".2f")
                st.plotly_chart(fig_attn, use_container_width=True)
            with col_out:
                st.subheader("Context Output Vectors ($A \\times V$)")
                df_ctx = pd.DataFrame(context_vectors, index=tokens, columns=[f"out_{i}" for i in range(d_k)])
                st.dataframe(df_ctx.style.highlight_max(axis=1), use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Example: PyTorch Scaled Dot-Product Self-Attention")
        st.code("""
import torch
import torch.nn.functional as F
import math

# Input Sequence Setup: Batch=1, Sequence Length=4, Model Dimension d_k=8
seq_len, d_k = 4, 8
X = torch.randn(1, seq_len, d_k)

# Projection Weight Matrices
W_q = torch.randn(d_k, d_k)
W_k = torch.randn(d_k, d_k)
W_v = torch.randn(d_k, d_k)

# 1. Linear Projection to Query, Key, Value spaces
Q = torch.matmul(X, W_q)
K = torch.matmul(X, W_k)
V = torch.matmul(X, W_v)

# 2. Scaled Dot-Product Scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

# 3. Softmax Normalization
attention_weights = F.softmax(scores, dim=-1)

# 4. Context Output via Weighted Mixture of Values
output = torch.matmul(attention_weights, V)

print("Attention Weight Matrix Shape:", attention_weights.shape) # [1, 4, 4]
print("Context Output Shape:", output.shape)                   # [1, 4, 8]
""", language="python")

    # -------------------------------------------------------------------------
    # TAB 2: BERT vs GPT
    # -------------------------------------------------------------------------
    with tab2:
        st.header("2. BERT (Encoder-Only) vs. GPT (Decoder-Only)")
        
        st.markdown("""
        While both use Transformers, they use fundamentally different **Attention Masking Strategies** depending on their pre-training objectives:
        """)

        col_b_info, col_g_info = st.columns(2)
        with col_b_info:
            st.success("""
            ### 🔵 BERT (Encoder-Only)
            * **Attention:** **Bidirectional** (Look left AND right).
            * **Pre-training:** Masked Language Modeling (Fill in `[MASK]` tokens).
            * **Best For:** Comprehension tasks (Classification, Search, Named Entity Recognition).
            """)
        with col_g_info:
            st.error("""
            ### 🔴 GPT (Decoder-Only)
            * **Attention:** **Causal / Unidirectional** (Look left ONLY).
            * **Pre-training:** Autoregressive Next-Token Prediction.
            * **Best For:** Generative tasks (Chat, Story Writing, Code Generation).
            """)

        st.markdown("---")
        arch_choice = st.radio("Select Attention Mask Visualizer:", ["BERT (Bidirectional Attention)", "GPT (Causal Masked Attention)"], horizontal=True)
        
        sample_sentence = "The student submitted the final exam"
        words = sample_sentence.split()
        N = len(words)
        
        if arch_choice == "BERT (Bidirectional Attention)":
            st.subheader("BERT Bidirectional Attention Visibility")
            st.info("💡 **Full Context Visibility:** Every token can attend to every other token in the sequence (Mask value = 1 everywhere).")
            mask = np.ones((N, N))
            df_mask = pd.DataFrame(mask, index=words, columns=words)
            fig_mask = px.imshow(df_mask, x=words, y=words, color_continuous_scale="Greens", text_auto=True)
            fig_mask.update_traces(showscale=False)
            st.plotly_chart(fig_mask, use_container_width=True)
        else:
            st.subheader("GPT Causal Masked Attention Visibility")
            st.warning("🔒 **Triangular Mask:** A token at index $i$ cannot attend to future tokens $j > i$. Future positions are set to $-\\infty$ before softmax, yielding zero probability.")
            causal_mask = np.tril(np.ones((N, N)))
            df_mask = pd.DataFrame(causal_mask, index=words, columns=words)
            fig_mask = px.imshow(df_mask, x=words, y=words, color_continuous_scale="Reds", text_auto=True)
            fig_mask.update_traces(showscale=False)
            st.plotly_chart(fig_mask, use_container_width=True)

        st.markdown("---")
        st.subheader("💻 Practical Code Example: PyTorch Causal Mask Generation")
        st.code("""
import torch

seq_len = 5

# Generate Lower-Triangular Causal Mask Matrix
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

# Create PyTorch Mask Tensor with -inf for Softmax
attn_scores = torch.zeros(seq_len, seq_len)
masked_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

print("Causal Softmax Input Mask Matrix:\n", masked_scores)
""", language="python")

    # -------------------------------------------------------------------------
    # TAB 3: LLM Generation Parameters
    # -------------------------------------------------------------------------
    with tab3:
        st.header("3. LLM Decoding & Sampling Hyperparameters")
        
        st.info("""
        💡 **How Generation Works:** At each step, an autoregressive LLM outputs raw scores called **Logits** for every token in its vocabulary. 
        Hyperparameters control how these raw logits are filtered and transformed into final probabilities before sampling.
        """)

        vocab_candidates = ["coffee", "tea", "water", "laptop", "banana", "galaxy", "elephant", "whisky"]
        raw_logits = np.array([5.2, 4.8, 3.5, 1.2, 0.8, -1.0, -2.5, 0.5])
        
        col_params, col_logits = st.columns([1, 1.2])
        with col_params:
            st.markdown("### 🎛️ Decoding Knobs")
            prompt_context = st.text_input("Prompt Context:", value="Every morning, I start my day with a hot cup of ...")
            
            temp = st.slider("Temperature ($T$):", 0.1, 2.0, 0.7, step=0.1,
                             help="T < 1 makes distribution sharper/confident; T > 1 flattens distribution for creativity.")
            top_k = st.slider("Top-$k$ Filtering:", 1, len(vocab_candidates), 4,
                              help="Keeps only the K highest-probability tokens.")
            top_p = st.slider("Top-$p$ (Nucleus) Filtering:", 0.1, 1.0, 0.9, step=0.05,
                              help="Keeps the smallest set of tokens whose cumulative probability exceeds p.")

        # Math Processing
        scaled_logits = raw_logits / temp
        tensor_logits = torch.tensor(scaled_logits, dtype=torch.float32)
        
        # Apply Top-k
        if top_k < len(vocab_candidates):
            topk_values, topk_indices = torch.topk(tensor_logits, top_k)
            mask = torch.full_like(tensor_logits, float('-inf'))
            mask.scatter_(0, topk_indices, topk_values)
            tensor_logits = mask
            
        # Apply Top-p
        probs = F.softmax(tensor_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        tensor_logits[indices_to_remove] = float('-inf')
        
        final_probs = F.softmax(tensor_logits, dim=-1).numpy()
        
        df_gen = pd.DataFrame({
            "Candidate Word": vocab_candidates,
            "Raw Logit": raw_logits,
            "Final Probability": final_probs
        }).sort_values(by="Final Probability", ascending=False)
        
        with col_logits:
            st.markdown("### 📊 Probability Distribution")
            fig_gen = px.bar(
                df_gen, x="Candidate Word", y="Final Probability", color="Final Probability",
                title=f"Adjusted Sampling Probabilities (T={temp}, Top-k={top_k}, Top-p={top_p})",
                color_continuous_scale="Purples"
            )
            fig_gen.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_gen, use_container_width=True)
            
            filtered_words = df_gen[df_gen["Final Probability"] > 0]["Candidate Word"].tolist()
            st.success(f"**Eligible Candidate Tokens for Generation:** `{filtered_words}`")

        st.markdown("---")
        st.subheader("💻 Practical Code Example: Temperature & Top-k Sampling in PyTorch")
        st.code("""
import torch
import torch.nn.functional as F

logits = torch.tensor([5.2, 4.8, 3.5, 1.2, 0.8, -1.0, -2.5, 0.5])
temperature = 0.7
top_k = 3

# 1. Apply Temperature Scaling (z_i / T)
scaled_logits = logits / temperature

# 2. Filter out all tokens beyond Top-K with -inf
topk_values, topk_indices = torch.topk(scaled_logits, top_k)
filtered_logits = torch.full_like(scaled_logits, float('-inf'))
filtered_logits.scatter_(0, topk_indices, topk_values)

# 3. Softmax to create valid probability distribution
probabilities = F.softmax(filtered_logits, dim=-1)

# 4. Sample next token index using multinomial sampling
sampled_index = torch.multinomial(probabilities, num_samples=1)

print("Final Token Probabilities:", probabilities.numpy().round(3))
print("Sampled Next Token Index:", sampled_index.item())
""", language="python")
