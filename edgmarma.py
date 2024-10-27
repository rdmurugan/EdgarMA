import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch.utils.tensorboard import SummaryWriter  # TensorBoard integration
import wget
# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Set up TensorBoard writer
writer = SummaryWriter('runs/m_a_analysis')  # Log directory

# Step 0: Set up headers for SEC requests with proper User-Agent


# Step 0: Set up headers for SEC requests with proper User-Agent
headers = {'User-Agent': 'Durai Rajamanickam(drajamanicka@ualr.edu) - Data for academic research purposes'}

# Step 1: Data Collection from EDGAR SEC
def fetch_filings(cik_list, count=2):
    filings = []
    for cik in cik_list:
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            docs = data.get('filings', {}).get('recent', {})
            accession_numbers = docs.get('accessionNumber', [])
            forms = docs.get('form', [])
            primary_docs = docs.get('primaryDocument', [])
            for i in range(len(accession_numbers)):
                if len(filings) >= count:
                    break
                if forms[i] == '10-K':
                    accession_number = accession_numbers[i].replace('-', '')
                    primary_doc = primary_docs[i]
                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_doc}"
                    filing_response = requests.get(filing_url, headers=headers)
                    if filing_response.status_code == 200:
                        filings.append(filing_response.text)
                    else:
                        print(f"Failed to fetch filing document for CIK {cik}")
        except Exception as e:
            print(f"Error fetching data for CIK {cik}: {e}")
    writer.add_scalar("Data/filings_fetched", len(filings))  # Log number of filings fetched
    return filings

# Example CIKs (Apple Inc. and Microsoft Corp.)
cik_list = [
    '0000320193',  # Apple Inc.
    '0000789019',  # Microsoft Corp.
    '0001652044',  # Alphabet Inc. (Google)
    '0001018724',  # Amazon.com Inc.
    '0001326801',  # Facebook Inc. (Meta)
    '0001318605',  # Tesla Inc.
    '0001067983',  # Berkshire Hathaway
    '0000200406',  # Johnson & Johnson
    '0000019617',  # JPMorgan Chase & Co.
    '0001403161',  # Visa Inc.
    '0000080424',  # Procter & Gamble
    '0000034088',  # Exxon Mobil Corp.
    '0000050863',  # Intel Corp.
    '0000104169',  # Walmart Inc.
    '0000858877',  # Cisco Systems Inc.
    '0000093410',  # Chevron Corp.
    '0000077476',  # PepsiCo Inc.
    '0000021344',  # The Coca-Cola Company
    '0000078003',  # Pfizer Inc.
    '0001467858',  # General Motors Co.
    '0000037996',  # Ford Motor Co.
    '0001065280',  # Netflix Inc.
    '0000796343',  # Adobe Inc.
    '0000051143'   # IBM Corp.
]


# Fetch a small number of filings (e.g., 2 filings total)
filings = fetch_filings(cik_list, count=24)

# Step 2: Preprocessing and Section Extraction
keywords = ['acquisition', 'merger', 'business combination', 'asset purchase', 'stock purchase', 'amalgamation', 'consolidation']

def preprocess_and_extract(filings, keywords):
    extracted_sections = []
    for filing in filings:
        soup = BeautifulSoup(filing, 'html.parser')
        text = soup.get_text(separator='\n')
        sections = text.split('\n')
        for section in sections:
            if any(keyword.lower() in section.lower() for keyword in keywords):
                extracted_sections.append(section.strip())
    writer.add_scalar("Data/sections_extracted", len(extracted_sections))  # Log number of extracted sections
    return extracted_sections

extracted_sections = preprocess_and_extract(filings, keywords)

# Step 3: Entity Recognition with General-Purpose NER Model
# Load a general-purpose NER model and tokenizer
model_name = 'dslim/bert-base-NER'  # General-purpose NER model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Use Hugging Face pipeline for NER
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def perform_ner_pipeline(sections):
    ner_results = []
    total_entities = 0
    for section in sections:
        truncated_section = section[:1000]  # Limit section length
        ner_output = nlp(truncated_section)
        entities = [(entity['word'], entity['entity_group']) for entity in ner_output]
        total_entities += len(entities)  # Count total entities
        ner_results.append({'section': section, 'entities': entities})
    writer.add_scalar("NER/total_entities", total_entities)  # Log total entities recognized
    return ner_results

ner_results = perform_ner_pipeline(extracted_sections)

# Step 4: Semantic Filtering with GloVe Embeddings
def load_glove_embeddings(glove_file='./glove.6B.100d.txt'):
    """Loads GloVe embeddings from a file.

    Downloads the file if it doesn't exist.
    """
    if not os.path.exists(glove_file):
        print(f"Downloading GloVe embeddings to {glove_file}...")
        glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        wget.download(glove_url)
        !unzip glove.6B.zip # Assuming unzip is available on your system

    embeddings_index = {}
    try:
        with open(glove_file, encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 101:
                    continue  # Skip invalid lines
                word = values[0]
                coeffs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coeffs
        return embeddings_index
    except FileNotFoundError:
        print(f"Error: GloVe embeddings file not found at {glove_file}. Please download and place it in the specified location.")
        return {}


# Ensure that the GloVe embeddings file is available in your working directory
embeddings_index = load_glove_embeddings()

# M&A-related terms embeddings
ma_terms = ['acquisition', 'merger', 'business combination', 'asset purchase', 'stock purchase', 'amalgamation', 'consolidation']
ma_embeddings = []
for term in ma_terms:
    words = term.lower().split()
    word_embeddings = [embeddings_index.get(word, None) for word in words if embeddings_index.get(word, None) is not None]
    if word_embeddings:
        term_embedding = np.mean(word_embeddings, axis=0)
        ma_embeddings.append(term_embedding)
ma_embeddings = np.array(ma_embeddings)

def semantic_filtering(ner_results, threshold=0.7):
    filtered_results = []
    for result in ner_results:
        filtered_entities = []
        for entity_text, entity_label in result['entities']:
            words = entity_text.lower().split()
            word_embeddings = [embeddings_index.get(word, None) for word in words]
            word_embeddings = [emb for emb in word_embeddings if emb is not None]
            if not word_embeddings:
                continue  # Skip if embeddings are missing
            entity_embedding = np.mean(word_embeddings, axis=0)
            similarities = cosine_similarity([entity_embedding], ma_embeddings)
            max_similarity = np.max(similarities)
            writer.add_scalar("SemanticFiltering/max_similarity", max_similarity)  # Log similarity scores
            if max_similarity >= threshold:
                filtered_entities.append((entity_text, entity_label))
        if filtered_entities:
            filtered_results.append({'section': result['section'], 'entities': filtered_entities})
    writer.add_scalar("SemanticFiltering/filtered_entities_count", len(filtered_results))  # Log number of filtered results
    return filtered_results

filtered_results = semantic_filtering(ner_results)

# Step 5: Contextual Filtering with Sentence Embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined M&A sentence patterns
ma_patterns = [
    "The Company entered into a Merger Agreement with",
    "We acquired all outstanding shares of",
    "On [Date], completed the acquisition of",
    "The Company has agreed to acquire",
    "We have entered into a definitive agreement to merge",
    "Pursuant to the merger agreement, the shareholders will receive",
    "An acquisition of substantial assets was completed",
    "The company finalized the acquisition of",
    "A merger with [Company Name] was completed",
    "The merger agreement was signed on [Date]",
    "The acquisition of [Company Name] was finalized",
    "We have completed the merger with",
    "A definitive agreement to acquire [Company Name]",
    "A merger or acquisition agreement was executed with"
]

ma_pattern_embeddings = sbert_model.encode(ma_patterns)

def contextual_filtering(filtered_results, threshold=0.2):
    final_entities = []
    for result in filtered_results:
        section = result['section']
        sentences = sent_tokenize(section)
        for sentence in sentences:
            sentence_embedding = sbert_model.encode(sentence)
            similarities = cosine_similarity([sentence_embedding], ma_pattern_embeddings)
            max_similarity = np.max(similarities)
            writer.add_scalar("ContextualFiltering/max_similarity", max_similarity)  # Log similarity scores in contextual filtering
            if max_similarity >= threshold:
                final_entities.append({'sentence': sentence, 'entities': result['entities']})
                break  # Assuming entities are relevant to the sentence
    writer.add_scalar("ContextualFiltering/final_entities_count", len(final_entities))  # Log number of final entities
    return final_entities

final_entities = contextual_filtering(filtered_results)
print (final_entities)
# Step 6: Displaying the Results
for item in final_entities:
    print(f"Sentence: {item['sentence']}")
    print("Entities:")
    for entity, label in item['entities']:
        print(f" - {entity}: {label}")
    print("-" * 80)

# Close the TensorBoard writer
writer.close()
